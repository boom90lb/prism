"""trend_v1 adjudication helper — fragility kill and convexity read (docs/trend_design.md §4).

Read-only over finished trial directories: opens no file for writing except
``--out``; appends to no ledger; moves no ratified value.

    python -m research.scripts.trend_adjudicate --t0 results/trend_t0 \\
        [--trials results/trend_t1,results/trend_t2,results/trend_t3,results/trend_t4] \\
        [--ledger results/trend_v1_trials.jsonl] [--b1_returns results/demotion_b1/returns.csv] \\
        [--oos_start 2026-07-20] [--decile 0.10] [--out path.json]

**Fragility** (readable only with all four trial dirs; else ``{"readable": false}``):
``S_k = summary.json["sharpe"]`` (net annualized = periodic x sqrt(252),
``src/prism/execution/target_weights.py:107-108``). Single-knob guard: the flattened
``config.json`` diff vs T0 must equal exactly ``{signal.lookback_bars}``,
``{signal.skip_bars}``, ``{signal.decision_every}``, ``{construction.sizing}`` by position
(pattern ``research/scripts/momentum_fragility_read.py:143-152``); every dir must carry
``price_return_fallback == False``, ``price_convention == "total_return"`` and an
``end_date`` resolving to 2026-07-17 NY midnight — a fallback run or a sub-sample run is
not a registered cell; the ledger ``output_dir`` sequence must start ``[t0, t1, t2, t3, t4]``.
kill (a): ``median(S_1..S_4) < 0`` with NaN mapped to ``-inf`` (a probe that produced no
Sharpe is adverse). kill (b): ``exists k: (S_k < 0) != (S_0 < 0) and |S_k| > |S_0|``, NaN
excluded. Secondary lens: the same flip test on ``total_return``.

**Convexity** (on T0; per sample in {full, oos}): decision bars ``d`` from ``folds.json``;
window ``W_d`` = the 21 stitched return rows at panel positions ``d+1..d+21`` (fill at open
``d+1``, held through the next decision bar's own return row); complete windows only.
``R^S_d = prod(1 + r_t) - 1`` over ``W_d`` (net, ``returns.csv``). Conditioning (a):
``R^M_d = SPY_close[d+21] / SPY_close[d] - 1`` from ``score_close.csv``. Conditioning (b):
``R^B_d = prod(1 + b_t) - 1`` over ``W_d`` from ``results/demotion_b1/returns.csv``
(``daily_return``, parsed ``utc=True -> tz_convert("America/New_York")``,
``research/scripts/joint_crash_receipt.py:44-50``), defined only when every row of ``W_d``
lies in the B1 stream — the separate ``b1_overlap`` leg. Per leg: ``m = max(1, floor(decile
* n))`` smallest ``R^c`` (stable sort, ties by ``d``), ``mu_cond`` = mean sleeve window return
over them, ``mu_unc`` over all defined windows, pass iff ``mu_cond - mu_unc >= 0``. The read
passes iff both legs pass. OOS sample = windows with ``d >= oos_start`` (empty for T0-T4).

**Readability of the convexity ``--t0`` dir.** The OOS segment is the T5 promotion read
(``docs/trend_design.md`` §3: "post-ratification OOS segment reported separately"), and the
only directory that can contain decision bars at or after ``oos_start`` is a T5 cell whose
``config.end_date`` is >= 2027-07-17. The convexity ``--t0`` argument therefore accepts a T0
shape (``end_date == 2026-07-17`` NY) OR a T5 shape (``end_date > 2026-07-17``), still
refusing anything earlier (a sub-sample); ``inputs.t0_end_date`` records which was read.
The fragility read stays strict: T0-T4 must all resolve to 2026-07-17.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.arbitrage.trend_walk_forward import TREND_END_DATE_DEFAULT, TREND_TZ, resolve_end_date

WINDOW_BARS = 21
# Registered probes relative to T0: (hashed key, registered value) — docs/trend_design.md §3.
TRIAL_CELLS: tuple[tuple[str, object], ...] = (
    ("signal.lookback_bars", 126),
    ("signal.skip_bars", 0),
    ("signal.decision_every", 63),
    ("construction.sizing", "equal_notional"),
)
TRIAL_KEYS: tuple[str, ...] = tuple(k for k, _ in TRIAL_CELLS)
# T0's pinned values on the same keys (plus the convention), asserted before any read.
T0_PINS: dict[str, object] = {
    "signal.lookback_bars": 252,
    "signal.skip_bars": 21,
    "signal.decision_every": 21,
    "construction.sizing": "inverse_vol",
    "price_convention": "total_return",
}
LEDGER_DEFAULT = "results/trend_v1_trials.jsonl"
TRIAL_LABELS: tuple[str, ...] = ("T1 lookback 126", "T2 skip 0", "T3 decision_every 63", "T4 equal_notional")
VERDICT_REFUSED = "positive-carry, non-convex; portfolio admission refused"
VERDICT_NONPOSITIVE = "non-convex (and non-positive carry)"
VERDICT_CONVEX = "convex: left wing holds on both conditioning sets"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    # momentum_fragility_read.py:66-73
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, sort_keys=True) if isinstance(v, list) else v
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _ny_index(raw: pd.Index) -> pd.DatetimeIndex:
    # joint_crash_receipt.py:44-50: tz-aware NY midnight strings, mixed offsets -> utc=True first.
    return pd.DatetimeIndex(pd.to_datetime(raw, utc=True)).tz_convert(TREND_TZ)


def _resolve_config_end(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(str(value))
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(TREND_TZ)
    return ts.tz_convert(TREND_TZ)


def registered_cell_problems(run_dir: Path, *, allow_t5: bool = False) -> list[str]:
    """Reasons a directory is NOT a registered cell (empty when it is).

    ``allow_t5=False`` (fragility, T0-T4): ``config.end_date`` must resolve to exactly
    2026-07-17 NY. ``allow_t5=True`` (the convexity ``--t0`` argument only): a T5 shape,
    ``end_date > 2026-07-17``, is also accepted — the post-ratification OOS segment
    (``docs/trend_design.md`` §3, T5) can only ever live in such a directory. A sub-sample
    directory (``end_date < 2026-07-17``) is refused under both. The fallback and
    convention checks are identical under both.
    """
    problems: list[str] = []
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return [f"{run_dir}: config.json missing"]
    cfg = _load_json(config_path)
    if cfg.get("price_return_fallback") is not False:
        problems.append(f"{run_dir}: price_return_fallback is {cfg.get('price_return_fallback')!r}, not False")
    if cfg.get("price_convention") != "total_return":
        problems.append(f"{run_dir}: price_convention is {cfg.get('price_convention')!r}, not total_return")
    end = _resolve_config_end(cfg.get("end_date"))
    end_default = resolve_end_date(TREND_END_DATE_DEFAULT)
    if end is None:
        problems.append(f"{run_dir}: config.end_date {cfg.get('end_date')!r} does not resolve to a date")
    elif allow_t5:
        if end < end_default:
            problems.append(
                f"{run_dir}: config.end_date {cfg.get('end_date')!r} is before {TREND_END_DATE_DEFAULT} NY "
                "(a sub-sample dir is neither T0 nor T5)"
            )
    elif end != end_default:
        problems.append(f"{run_dir}: config.end_date {cfg.get('end_date')!r} does not resolve to {TREND_END_DATE_DEFAULT} NY")
    return problems


def config_end_date_text(run_dir: Path) -> str | None:
    """``config.json["end_date"]`` of a run dir as written (None when absent), for the report's inputs."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    value = _load_json(config_path).get("end_date")
    return None if value is None else str(value)


# --------------------------------------------------------------------------------------
# Fragility
# --------------------------------------------------------------------------------------


def fragility_arithmetic(
    s0: float,
    s_trials: Sequence[float],
    r0: float,
    r_trials: Sequence[float],
) -> dict[str, Any]:
    """kill (a) median with NaN -> -inf; kill (b) sign flip at |S_k| > |S_0|, NaN excluded; secondary lens."""
    sharpes = [_as_float(s) for s in s_trials]
    returns = [_as_float(r) for r in r_trials]
    s0 = _as_float(s0)
    r0 = _as_float(r0)
    for_median = [(-math.inf if math.isnan(s) else s) for s in sharpes]
    median_sharpe = statistics.median(for_median) if for_median else float("nan")
    flips_sharpe = [
        k + 1
        for k, s in enumerate(sharpes)
        if not math.isnan(s) and not math.isnan(s0) and (s < 0) != (s0 < 0) and abs(s) > abs(s0)
    ]
    flips_return = [
        k + 1
        for k, r in enumerate(returns)
        if not math.isnan(r) and not math.isnan(r0) and (r < 0) != (r0 < 0) and abs(r) > abs(r0)
    ]
    kill_a = bool(median_sharpe < 0)
    kill_b = bool(flips_sharpe)
    return {
        "rule": "docs/trend_design.md §4 fragility kill: median(net annualized Sharpe of T1-T4) < 0 (NaN -> -inf), "
        "or any single probe flips the sign of the net result at magnitude > T0's point estimate (NaN excluded)",
        "t0_sharpe": s0,
        "t0_total_return": r0,
        "trial_sharpes": sharpes,
        "trial_total_returns": returns,
        "median_sharpe_t1_t4": median_sharpe,
        "n_nan_sharpes": int(sum(math.isnan(s) for s in sharpes)),
        "kill_a_median_negative": kill_a,
        "kill_b_sign_flip_sharpe": kill_b,
        "kill_b_flipping_trials_sharpe": flips_sharpe,
        "secondary_sign_flip_total_return": flips_return,
        "verdict": "KILL (fragility)" if (kill_a or kill_b) else "SURVIVES fragility read",
        "note": "Survival is not promotion: promotion reads only at T5 (docs/trend_design.md §3-4).",
    }


def _single_knob_problems(t0_dir: Path, trial_dirs: Sequence[Path]) -> list[str]:
    """T0 carries the pinned values; each probe moves exactly its registered key to its registered value."""
    a = _flat(_load_json(t0_dir / "config.json"))
    problems: list[str] = [
        f"{t0_dir}: T0 config {k}={a.get(k)!r} != pinned {v!r}" for k, v in T0_PINS.items() if a.get(k) != v
    ]
    for (key, value), run_dir in zip(TRIAL_CELLS, trial_dirs):
        b = _flat(_load_json(run_dir / "config.json"))
        moved = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        if moved != {key}:
            problems.append(f"{run_dir}: config keys moved vs T0 {sorted(moved)} != registered {[key]}")
        elif b.get(key) != value:
            problems.append(f"{run_dir}: {key}={b.get(key)!r} != registered value {value!r}")
    return problems


def _ledger_order_problems(ledger_path: Path | None, t0_dir: Path, trial_dirs: Sequence[Path]) -> list[str]:
    if ledger_path is None:
        return ["the fragility read requires --ledger (the family ledger's output_dir sequence is part of the read)"]
    if not ledger_path.is_file():
        return [f"ledger {ledger_path} not found"]
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(ledger_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            return [f"ledger {ledger_path} line {lineno} is corrupt; refusing to read the sequence"]
    actual = [str(Path(str(r.get("output_dir"))).resolve()) for r in rows]
    expected = [str(Path(p).resolve()) for p in (t0_dir, *trial_dirs)]
    if actual[: len(expected)] != expected:
        return [f"ledger output_dirs {actual} do not start with the registered sequence {expected}"]
    return []


def fragility_read(t0_dir: Path, trial_dirs: Sequence[Path], ledger_path: Path | None) -> dict[str, Any]:
    if len(trial_dirs) != 4:
        return {"readable": False, "reason": f"fragility needs exactly T1-T4, got {len(trial_dirs)} trial dirs"}
    problems: list[str] = []
    for run_dir in (t0_dir, *trial_dirs):
        for name in ("summary.json", "config.json"):
            if not (run_dir / name).is_file():
                problems.append(f"{run_dir}: {name} missing")
    if problems:
        return {"readable": False, "reason": "; ".join(problems)}
    for run_dir in (t0_dir, *trial_dirs):
        problems.extend(registered_cell_problems(run_dir))
    if problems:
        return {"readable": False, "reason": "; ".join(problems)}
    problems.extend(_single_knob_problems(t0_dir, trial_dirs))
    problems.extend(_ledger_order_problems(ledger_path, t0_dir, trial_dirs))
    if problems:
        return {"readable": False, "reason": "; ".join(problems)}
    s0_summary = _load_json(t0_dir / "summary.json")
    trial_summaries = [_load_json(d / "summary.json") for d in trial_dirs]
    read = fragility_arithmetic(
        _as_float(s0_summary.get("sharpe")),
        [_as_float(s.get("sharpe")) for s in trial_summaries],
        _as_float(s0_summary.get("total_return")),
        [_as_float(s.get("total_return")) for s in trial_summaries],
    )
    read["readable"] = True
    read["trials"] = [
        {"run_dir": str(d), "delta": label, "config_hash": s.get("config_hash")}
        for d, label, s in zip(trial_dirs, TRIAL_LABELS, trial_summaries)
    ]
    read["t0"] = {"run_dir": str(t0_dir), "config_hash": s0_summary.get("config_hash")}
    return read


# --------------------------------------------------------------------------------------
# Convexity
# --------------------------------------------------------------------------------------


def _compound(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.prod(1.0 + arr) - 1.0)


def convexity_windows(
    returns: pd.Series,
    panel_index: pd.DatetimeIndex,
    spy_close: pd.Series,
    decision_bars: Sequence[pd.Timestamp],
    b1: pd.Series | None,
    *,
    window: int = WINDOW_BARS,
) -> list[dict[str, Any]]:
    """One record per complete window: ``{"d", "sleeve", "spy", "b1"}`` (NaN where undefined)."""
    pos_of = {ts: i for i, ts in enumerate(panel_index)}
    ret_index = returns.index
    b1_index = None if b1 is None else b1.index
    rows: list[dict[str, Any]] = []
    for d in decision_bars:
        p = pos_of.get(d)
        if p is None or p + window >= len(panel_index):
            continue
        w_dates = panel_index[p + 1 : p + window + 1]
        if not w_dates.isin(ret_index).all():
            continue
        sleeve = _compound(returns.loc[w_dates])
        spy0 = float(spy_close.iloc[p])
        spy1 = float(spy_close.iloc[p + window])
        spy = spy1 / spy0 - 1.0 if np.isfinite(spy0) and np.isfinite(spy1) and spy0 > 0.0 else float("nan")
        b1_ret = float("nan")
        if b1 is not None and b1_index is not None and w_dates.isin(b1_index).all():
            b1_ret = _compound(b1.loc[w_dates])
        rows.append({"d": pd.Timestamp(d), "sleeve": sleeve, "spy": spy, "b1": b1_ret})
    return rows


def conditional_leg(rows: Sequence[dict[str, Any]], key: str, decile: float) -> dict[str, Any] | None:
    """Worst-``decile`` conditioning of the sleeve's window return on ``rows[key]``."""
    defined = [r for r in rows if np.isfinite(r[key]) and np.isfinite(r["sleeve"])]
    n = len(defined)
    if n == 0:
        return None
    m = max(1, int(math.floor(decile * n)))
    ordered = sorted(defined, key=lambda r: r[key])  # stable: ties keep d order
    cond = ordered[:m]
    mu_cond = float(np.mean([r["sleeve"] for r in cond]))
    mu_unc = float(np.mean([r["sleeve"] for r in defined]))
    diff = mu_cond - mu_unc
    out: dict[str, Any] = {
        "n": n,
        "m": m,
        "threshold": float(cond[-1][key]),
        "worst_decision_bars": [r["d"].isoformat() for r in cond],
        "mu_cond": mu_cond,
        "mu_unc": mu_unc,
        "diff": diff,
        "pass": bool(diff >= 0.0),
        "span": [defined[0]["d"].isoformat(), defined[-1]["d"].isoformat()],
    }
    if m == 1:
        out["power_note"] = "m == 1: the conditional mean is a single window; no power to speak of"
    return out


def convexity_sample(rows: Sequence[dict[str, Any]], decile: float) -> dict[str, Any]:
    spy = conditional_leg(rows, "spy", decile)
    b1 = conditional_leg(rows, "b1", decile)
    out: dict[str, Any] = {"n_windows": len(rows), "spy": spy, "b1_overlap": b1}
    if spy is None or b1 is None:
        out["pass"] = None
        out["reason"] = "a conditioning leg has no defined window" + (
            " (no B1 overlap)" if b1 is None and spy is not None else ""
        )
    else:
        out["pass"] = bool(spy["pass"] and b1["pass"])
    return out


def convexity_read(
    returns: pd.Series,
    panel_index: pd.DatetimeIndex,
    spy_close: pd.Series,
    decision_bars: Sequence[pd.Timestamp],
    b1: pd.Series | None,
    *,
    decile: float = 0.10,
    oos_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if not 0.0 < decile <= 0.5:
        raise ValueError(f"decile must be in (0, 0.5], got {decile}")
    rows = convexity_windows(returns, panel_index, spy_close, decision_bars, b1)
    full = convexity_sample(rows, decile)
    oos: dict[str, Any]
    if oos_start is None:
        oos = {"n_windows": 0, "pass": None, "reason": "no oos_start given"}
    else:
        oos_rows = [r for r in rows if r["d"] >= oos_start]
        if oos_rows:
            oos = convexity_sample(oos_rows, decile)
        else:
            oos = {
                "n_windows": 0,
                "pass": None,
                "reason": f"no decision bar at or after oos_start {oos_start.isoformat()} (empty for T0-T4)",
            }
    return {"decile": float(decile), "window_bars": WINDOW_BARS, "full_sample": full, "oos_segment": oos}


def convexity_verdict(full_pass: bool | None, s0: float) -> str:
    if full_pass is None:
        return "unreadable: a conditioning leg has no defined window"
    if not full_pass:
        return VERDICT_REFUSED if s0 > 0 else VERDICT_NONPOSITIVE
    return VERDICT_CONVEX


# --------------------------------------------------------------------------------------
# Directory loading (read-only)
# --------------------------------------------------------------------------------------


def load_returns_csv(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    frame.index = _ny_index(frame.index)
    col = "daily_return" if "daily_return" in frame.columns else frame.columns[0]
    return frame[col].astype(float)


def load_score_close_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = _ny_index(frame.index)
    return frame.astype(float)


def load_decision_bars(folds_path: Path) -> list[pd.Timestamp]:
    folds = json.loads(folds_path.read_text())
    bars: list[pd.Timestamp] = []
    for fold in folds:
        for text in fold.get("decision_bars", []):
            bars.append(pd.Timestamp(text).tz_convert(TREND_TZ))
    return bars


def convexity_read_dir(
    t0_dir: Path,
    b1_returns: Path | None,
    *,
    decile: float,
    oos_start: pd.Timestamp,
) -> dict[str, Any]:
    returns = load_returns_csv(t0_dir / "returns.csv")
    score_close = load_score_close_csv(t0_dir / "score_close.csv")
    if "SPY" not in score_close.columns:
        raise SystemExit(f"{t0_dir}/score_close.csv has no SPY column (the US broad-equity bucket member)")
    decision_bars = load_decision_bars(t0_dir / "folds.json")
    b1 = load_returns_csv(b1_returns) if b1_returns is not None and b1_returns.is_file() else None
    out = convexity_read(
        returns,
        pd.DatetimeIndex(score_close.index),
        score_close["SPY"],
        decision_bars,
        b1,
        decile=decile,
        oos_start=oos_start,
    )
    out["b1_stream"] = (
        None
        if b1 is None
        else {"path": str(b1_returns), "span": [b1.index[0].isoformat(), b1.index[-1].isoformat()], "n": int(len(b1))}
    )
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--t0", required=True)
    p.add_argument("--trials", default=None, help="Comma-separated T1,T2,T3,T4 dirs (fragility read).")
    p.add_argument("--ledger", default=LEDGER_DEFAULT)
    p.add_argument("--b1_returns", default="results/demotion_b1/returns.csv")
    p.add_argument("--oos_start", default="2026-07-20")
    p.add_argument("--decile", type=float, default=0.10)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    t0_dir = Path(args.t0)
    trial_dirs = [Path(t) for t in args.trials.split(",") if t] if args.trials else []
    ledger_path = Path(args.ledger) if args.ledger else None
    oos_start = resolve_end_date(args.oos_start)
    s0 = _as_float(_load_json(t0_dir / "summary.json").get("sharpe")) if (t0_dir / "summary.json").is_file() else float("nan")
    fragility = (
        fragility_read(t0_dir, trial_dirs, ledger_path)
        if trial_dirs
        else {"readable": False, "reason": "no --trials given (fragility is readable only after T1-T4)"}
    )
    # The convexity --t0 dir may be T0 (2026-07-17) or the T5 extension (>= 2027-07-17, the only dir that
    # can carry a non-empty post-ratification OOS segment); fragility_read above stays strict (T0-T4 only).
    t0_problems = registered_cell_problems(t0_dir, allow_t5=True)
    if t0_problems:
        convexity: dict[str, Any] = {"readable": False, "reason": "; ".join(t0_problems)}
        verdict = "unreadable: T0 dir is not a registered cell"
    else:
        b1_path = Path(args.b1_returns) if args.b1_returns else None
        convexity = convexity_read_dir(t0_dir, b1_path, decile=float(args.decile), oos_start=oos_start)
        convexity["readable"] = True
        verdict = convexity_verdict(convexity["full_sample"]["pass"], s0)
    report = {
        "inputs": {
            "t0": str(t0_dir),
            "t0_end_date": config_end_date_text(t0_dir),
            "trials": [str(d) for d in trial_dirs],
            "ledger": str(ledger_path) if ledger_path else None,
            "b1_returns": args.b1_returns,
            "oos_start": oos_start.isoformat(),
            "decile": float(args.decile),
            "t0_sharpe": s0,
        },
        "fragility": fragility,
        "convexity": convexity,
        "verdict": verdict,
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
    print(text)
    return report


if __name__ == "__main__":
    main()
