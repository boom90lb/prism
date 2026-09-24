"""Public factor lab (registration ``registrations/2026-09-factor-lab.toml``).

Design and verdict rules: ``docs/public_factor_lab_design.md``. Three parts:

* ``priors``   uncounted: full-sample and post-publication Sharpe of public
  long-short series, the SPEC §4 publication bar and prior, correlations;
* ``fidelity`` uncounted: correlation of in-house books with the public series
  they implement (SPEC §4 criterion 2);
* ``sizing``   the counted selection set ``sizing_lab_v1``: four sizing rules on
  two public momentum series, one trial-log row per arm.

Every parameter is read from the registration TOML. Evidence runs refuse to
start unless the registration and design doc are committed unchanged at HEAD
(SPEC §4: register, then run).

Run from the repo root::

    python -m research.scripts.factor_lab --part all
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import tomllib
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.validation.compare import SharpeBootstrapSpec, sharpe_difference_interval
from prism.validation.metrics import deflated_sharpe_ratio, periodic_sharpe
from prism.validation.trials import (
    FINISHED,
    STARTED,
    RunRecorder,
    TrialLog,
    current_git_state,
    stable_json_dumps,
)

REGISTRATION = Path("registrations/2026-09-factor-lab.toml")
DESIGN_DOC = Path("docs/public_factor_lab_design.md")
CACHE_DIR = Path("data/public")
OUT_DIR = Path("results/2026-09-factor-lab")
TRIAL_LOG = Path("trials/sizing_lab_v1.jsonl")
USER_AGENT = "prism-factor-lab/1.0 (research; single-user)"
FRENCH_MISSING = (-99.99, -999.0)


class LabError(RuntimeError):
    """A lab input or invariant failed; raised, never swallowed (N7)."""


# --------------------------------------------------------------------- registration


def load_registration(path: Path = REGISTRATION) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def require_committed(paths: Sequence[Path]) -> str:
    """Refuse unless every path is tracked and unchanged against HEAD; return HEAD."""
    for path in paths:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], capture_output=True)
        if tracked.returncode != 0:
            raise LabError(f"{path} is not committed; the registration must precede the first run (SPEC §4)")
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)])
        if clean.returncode != 0:
            raise LabError(f"{path} differs from HEAD; an edited registration is a new registration (SPEC §4)")
    state = current_git_state()
    if not state.commit:
        raise LabError("git HEAD is unavailable; a run must carry its code commit")
    return str(state.commit)


# --------------------------------------------------------------------- data


def fetch_sources(sources: Mapping[str, Mapping[str, Any]], cache: Path, refresh: bool = False) -> dict[str, dict]:
    """Download each source zip once into ``cache``; return path, sha256, fetch time."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    manifest: dict[str, dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    out: dict[str, dict] = {}
    for source_id, spec in sources.items():
        target = cache / f"{source_id}.zip"
        entry = manifest.get(source_id)
        if refresh or not target.exists() or entry is None or entry.get("url") != spec["url"]:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = response.read()
            except Exception as exc:  # N7: a failed fetch raises with its source named
                raise LabError(f"fetch failed for {source_id} ({spec['url']}): {exc}") from exc
            if not payload:
                raise LabError(f"fetch returned no bytes for {source_id}")
            target.write_bytes(payload)
            entry = {"url": spec["url"], "fetched_at": datetime.now(timezone.utc).isoformat()}
        payload = target.read_bytes()
        entry = dict(entry)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["bytes"] = len(payload)
        entry["path"] = str(target)
        manifest[source_id] = entry
        out[source_id] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out


def _single_csv(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise LabError(f"expected one CSV in the archive, found {names}")
        return archive.read(names[0]).decode("utf-8", errors="replace")


_FRENCH_ROW = re.compile(r"^\s*(\d{8})\s*,")


def parse_french(payload: bytes) -> tuple[pd.DataFrame, str | None]:
    """First daily section of a Kenneth French CSV zip, in decimal units.

    Returns the frame (DatetimeIndex, stripped column names) and the CRSP
    vintage line when the file states one.
    """
    lines = _single_csv(payload).splitlines()
    vintage = next((line.strip() for line in lines if "CRSP database" in line), None)
    start = next((i for i, line in enumerate(lines) if _FRENCH_ROW.match(line)), None)
    if start is None or start == 0:
        raise LabError("no daily data rows found in the French file")
    header = [name.strip() for name in lines[start - 1].split(",")]
    rows: list[list[str]] = []
    for line in lines[start:]:
        if not _FRENCH_ROW.match(line):
            break
        rows.append([cell.strip() for cell in line.split(",")])
    frame = pd.DataFrame(rows, columns=["date", *header[1:]])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.set_index("date").apply(pd.to_numeric, errors="raise")
    for code in FRENCH_MISSING:
        frame = frame.mask(np.isclose(frame, code))
    return frame / 100.0, vintage


def parse_jkp(payload: bytes) -> pd.DataFrame:
    """A JKP factor CSV zip as a wide frame: DatetimeIndex × factor name, decimal returns."""
    frame = pd.read_csv(io.StringIO(_single_csv(payload)))
    for column in ("date", "name", "ret"):
        if column not in frame.columns:
            raise LabError(f"JKP file lacks column {column!r}: {list(frame.columns)}")
    frame["date"] = pd.to_datetime(frame["date"])
    wide = frame.pivot_table(index="date", columns="name", values="ret", aggfunc="first")
    return wide.sort_index()


def require_contiguous(series: pd.Series, name: str) -> pd.Series:
    """Trim to the first/last valid value and raise on any gap inside (N7)."""
    valid = series.dropna()
    if valid.empty:
        raise LabError(f"series {name} has no observations")
    inner = series.loc[valid.index[0] : valid.index[-1]]
    if inner.isna().any():
        gaps = inner[inner.isna()].index[:5].strftime("%Y-%m-%d").tolist()
        raise LabError(f"series {name} has missing values inside its range, first at {gaps}")
    return inner.astype(float)


@dataclass(frozen=True)
class PublicData:
    series: dict[str, pd.Series]  # registered series ids and "jkp:<theme>"
    market_total: pd.Series  # Mkt-RF + RF, decimal daily
    jkp_large_monthly: dict[str, pd.Series]
    vintages: dict[str, str | None]


def build_public_data(reg: Mapping[str, Any], fetched: Mapping[str, Mapping[str, Any]]) -> PublicData:
    raw_french: dict[str, pd.DataFrame] = {}
    raw_jkp: dict[str, pd.DataFrame] = {}
    vintages: dict[str, str | None] = {}
    for source_id, spec in reg["sources"].items():
        payload = Path(fetched[source_id]["path"]).read_bytes()
        if spec["format"] == "french":
            raw_french[source_id], vintages[source_id] = parse_french(payload)
        elif spec["format"] == "jkp":
            raw_jkp[source_id] = parse_jkp(payload)
            vintages[source_id] = None
        else:
            raise LabError(f"unknown source format {spec['format']!r}")

    def column(source_id: str, name: str) -> pd.Series:
        table = raw_french.get(source_id, raw_jkp.get(source_id))
        if table is None or name not in table.columns:
            raise LabError(f"source {source_id} has no column {name!r}")
        return table[name]

    series: dict[str, pd.Series] = {}
    for entry in reg["series"]:
        series[entry["id"]] = require_contiguous(column(entry["source"], entry["column"]), entry["id"])
    themes = reg["jkp_themes"]
    for name in themes["names"]:
        series[f"jkp:{name}"] = require_contiguous(column(themes["source"], name), f"jkp:{name}")
    ff3 = raw_french["ff3_daily"]
    market_total = require_contiguous(ff3["Mkt-RF"] + ff3["RF"], "MKT_RF+RF")
    large = {f"jkp_large:{name}": require_contiguous(raw_jkp["jkp_resmom_large_monthly"][name], name)
             for name in raw_jkp["jkp_resmom_large_monthly"].columns}
    return PublicData(series=series, market_total=market_total, jkp_large_monthly=large, vintages=vintages)


# --------------------------------------------------------------------- statistics


def annualized_stats(returns: pd.Series, per_year: int = 252) -> dict[str, Any]:
    values = returns.dropna().to_numpy(dtype=float)
    n = int(values.size)
    if n < 2:
        return {"n": n, "years": n / per_year, "sharpe": None, "t": None, "mean": None, "vol": None}
    sharpe_p = periodic_sharpe(values)
    sharpe = sharpe_p * math.sqrt(per_year) if np.isfinite(sharpe_p) else None
    years = n / per_year
    return {
        "n": n,
        "years": years,
        "sharpe": sharpe,
        "t": None if sharpe is None else sharpe * math.sqrt(years),
        "mean": float(values.mean() * per_year),
        "vol": float(values.std(ddof=1) * math.sqrt(per_year)),
        "first": str(returns.dropna().index[0].date()),
        "last": str(returns.dropna().index[-1].date()),
    }


def compound(returns: pd.Series | pd.DataFrame, rule: str) -> pd.Series | pd.DataFrame:
    """Compound daily returns to ``rule`` periods; a period with any missing day is NaN."""
    summed = np.log1p(returns).resample(rule).sum(min_count=1)
    missing = returns.isna().resample(rule).sum()
    return np.expm1(summed).mask(missing > 0)


def prior_record(returns: pd.Series, publication: str | None, cfg: Mapping[str, Any]) -> dict[str, Any]:
    per_year = int(cfg["sessions_per_year"])
    full = annualized_stats(returns, per_year)
    record: dict[str, Any] = {"full": full, "publication": publication}
    bar = (
        full["t"] is not None
        and full["t"] >= float(cfg["publication_bar_t"])
        and full["years"] >= float(cfg["publication_bar_min_years"])
    )
    record["publication_bar_pass"] = bool(bar)
    if publication is None:
        record.update({"pre": None, "post": None, "prior_sharpe": None, "prior_basis": "descriptive_only"})
        return record
    cut = pd.Timestamp(publication + "-01")
    pre = annualized_stats(returns[returns.index < cut], per_year)
    post = annualized_stats(returns[returns.index >= cut], per_year)
    record["pre"], record["post"] = pre, post
    if post["sharpe"] is not None and post["years"] >= float(cfg["post_publication_min_years"]):
        record["prior_sharpe"], record["prior_basis"] = post["sharpe"], "post_publication"
    elif full["sharpe"] is not None:
        record["prior_sharpe"] = float(cfg["pre_publication_haircut"]) * full["sharpe"]
        record["prior_basis"] = "haircut_full_sample"
    else:
        record["prior_sharpe"], record["prior_basis"] = None, "undefined"
    record["prior_pass"] = bool(bar and record["prior_sharpe"] is not None and record["prior_sharpe"] > 0)
    return record


def run_priors(reg: Mapping[str, Any], data: PublicData, out: Path) -> dict[str, Any]:
    cfg = reg["priors"]
    publications = {entry["id"]: entry.get("publication") for entry in reg["series"]}
    priors = {sid: prior_record(series, publications.get(sid), cfg) for sid, series in data.series.items()}
    frame = pd.DataFrame(data.series)
    (out / "correlations").mkdir(parents=True, exist_ok=True)
    correlation_files = []
    for start, end in cfg["correlation_windows"]:
        window = frame.loc[start:end]
        for frequency in cfg["correlation_frequencies"]:
            if frequency == "daily":
                table = window
            elif frequency == "monthly":
                table = compound(window, "ME")
            else:
                raise LabError(f"unknown correlation frequency {frequency!r}")
            corr = table.corr(min_periods=24)
            name = f"corr_{start[:4]}_{end[:4]}_{frequency}.csv"
            corr.round(6).to_csv(out / "correlations" / name)
            correlation_files.append(name)
    result = {"priors": priors, "correlation_files": correlation_files}
    (out / "priors.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


# --------------------------------------------------------------------- fidelity


def load_in_house(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    if "daily_return" not in frame.columns:
        raise LabError(f"{path} has no daily_return column")
    index = pd.to_datetime(frame.index, utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.Series(frame["daily_return"].to_numpy(dtype=float), index=index, name=path.stem)


def pair_stats(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    n = int(len(joined))
    if n < 3:
        return {"n": n, "corr": None, "beta": None}
    a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    corr = float(np.corrcoef(a, b)[0, 1])
    beta = float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    return {"n": n, "first": str(joined.index[0].date()), "last": str(joined.index[-1].date()), "corr": corr,
            "beta": beta}


def run_fidelity(reg: Mapping[str, Any], data: PublicData, priors: Mapping[str, Any], out: Path) -> dict[str, Any]:
    settings = reg["fidelity_settings"]
    records = []
    for entry in reg["fidelity"]:
        book = load_in_house(Path(entry["in_house"]))
        for reference in entry["references"]:
            if reference not in data.series:
                raise LabError(f"fidelity reference {reference!r} is not a loaded series")
            ref = data.series[reference]
            overlap = pd.concat([book, ref], axis=1, join="inner").dropna()
            daily = pair_stats(overlap.iloc[:, 0], overlap.iloc[:, 1])
            weekly = pair_stats(compound(overlap.iloc[:, 0], settings["weekly_anchor"]),
                                compound(overlap.iloc[:, 1], settings["weekly_anchor"]))
            prior = priors.get(reference, {}).get("prior_sharpe")
            implied = None if (prior is None or daily["corr"] is None) else daily["corr"] * prior
            records.append({
                "book": entry["label"],
                "reference": reference,
                "frequency": "daily",
                "daily": daily,
                "weekly": weekly,
                "book_sharpe_overlap": annualized_stats(overlap.iloc[:, 0])["sharpe"],
                "reference_sharpe_overlap": annualized_stats(overlap.iloc[:, 1])["sharpe"],
                "reference_prior_sharpe": prior,
                "implied_tracking_sharpe": implied,
                "meets_floor": None if daily["corr"] is None else daily["corr"] >= float(settings["floor"]),
            })
        for reference in entry.get("monthly_references", []):
            ref = data.jkp_large_monthly.get(reference)
            if ref is None:
                raise LabError(f"monthly fidelity reference {reference!r} is not loaded")
            book_monthly = compound(book, "ME").dropna()
            ref_monthly = ref.copy()
            ref_monthly.index = ref_monthly.index + pd.offsets.MonthEnd(0)
            records.append({"book": entry["label"], "reference": reference, "frequency": "monthly",
                            "monthly": pair_stats(book_monthly, ref_monthly)})
    result = {"floor": settings["floor"], "records": records}
    (out / "fidelity.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


# --------------------------------------------------------------------- sizing rules


def _effective(decisions: np.ndarray) -> np.ndarray:
    """Scale decided at close ``i`` applies from session ``i + 1`` (N2); 1 before any decision."""
    shifted = np.concatenate([[np.nan], decisions[:-1]])
    return pd.Series(shifted).ffill().fillna(1.0).to_numpy()


def refresh_mask(n: int, every: int) -> np.ndarray:
    return (np.arange(n) % int(every)) == 0


def scale_none(returns: np.ndarray, **_: Any) -> np.ndarray:
    return np.ones(returns.size)


def scale_constant_vol(returns: np.ndarray, *, refresh: int, target_vol: float, lookback_sessions: int,
                       max_scale: float, per_year: int = 252, **_: Any) -> np.ndarray:
    decisions = np.full(returns.size, np.nan)
    for i in np.flatnonzero(refresh_mask(returns.size, refresh)):
        if i + 1 < lookback_sessions:
            continue
        window = returns[i + 1 - lookback_sessions : i + 1]
        sigma = float(np.std(window, ddof=1) * math.sqrt(per_year))
        if not np.isfinite(sigma) or sigma <= 0:
            raise LabError(f"degenerate volatility estimate at session {i}")
        decisions[i] = min(max_scale, target_vol / sigma)
    return _effective(decisions)


def scale_variance_managed(returns: np.ndarray, *, refresh: int, lookback_sessions: int,
                           min_history_refreshes: int, max_scale: float, **_: Any) -> np.ndarray:
    decisions = np.full(returns.size, np.nan)
    history: list[float] = []
    for i in np.flatnonzero(refresh_mask(returns.size, refresh)):
        if i + 1 < lookback_sessions:
            continue
        window = returns[i + 1 - lookback_sessions : i + 1]
        rv = float(np.sum(window**2))
        if not np.isfinite(rv) or rv <= 0:
            raise LabError(f"degenerate realized variance at session {i}")
        history.append(rv)
        if len(history) >= min_history_refreshes:
            decisions[i] = min(max_scale, float(np.mean(history)) / rv)
    return _effective(decisions)


def crash_state(market_total: np.ndarray, trailing: int, theta: float) -> np.ndarray:
    """S_t: compounded market return over the ``trailing`` sessions ending t-1 is below θ.

    Sessions without ``trailing`` prior returns carry S = 0.
    """
    log_market = np.log1p(market_total)
    csum = np.concatenate([[0.0], np.cumsum(log_market)])  # csum[k] = sum of log returns [0, k)
    state = np.zeros(market_total.size, dtype=bool)
    for t in range(trailing, market_total.size):
        trailing_return = math.expm1(csum[t] - csum[t - trailing])  # sessions t-trailing .. t-1
        state[t] = trailing_return < theta
    return state


def scale_crash_degross(returns: np.ndarray, *, market_total: np.ndarray, refresh: int, theta: float,
                        trailing_sessions: int, multiplier: float, exit_clean_sessions: int,
                        **_: Any) -> np.ndarray:
    if market_total.size != returns.size:
        raise LabError("market series is not aligned with the base series")
    state = crash_state(market_total, trailing_sessions, theta)
    is_refresh = refresh_mask(returns.size, refresh)
    decisions = np.full(returns.size, np.nan)
    degrossed = False
    clean_run = 0
    for t in range(returns.size):
        clean_run = 0 if state[t] else clean_run + 1
        if not degrossed and state[t]:
            degrossed = True
            decisions[t] = multiplier
        elif degrossed and is_refresh[t] and clean_run >= exit_clean_sessions:
            degrossed = False
            decisions[t] = 1.0
    return _effective(decisions)


RULES: dict[str, Callable[..., np.ndarray]] = {
    "none": scale_none,
    "crash_degross_v1": scale_crash_degross,
    "constant_vol_bsc": scale_constant_vol,
    "variance_managed_mm": scale_variance_managed,
}


def apply_scale(returns: np.ndarray, scale: np.ndarray, *, base_gross: float, change_cost_bps: float,
                base_cost_per_gross_year: float, per_year: int = 252) -> tuple[np.ndarray, np.ndarray]:
    """Net daily returns and per-session cost for a scaled self-financing factor (N3)."""
    previous = np.concatenate([[1.0], scale[:-1]])
    change_cost = np.abs(scale - previous) * base_gross * change_cost_bps / 1e4
    base_cost = scale * base_gross * base_cost_per_gross_year / per_year
    cost = change_cost + base_cost
    return scale * returns - cost, cost


def drawdown_stats(net: pd.Series) -> dict[str, Any]:
    equity = (1.0 + net).cumprod()
    drawdown = 1.0 - equity / equity.cummax()
    rolling = np.expm1(np.log1p(net).rolling(21).sum())
    yearly = np.expm1(np.log1p(net).groupby(net.index.year).sum())
    return {
        "max_drawdown": float(drawdown.max()),
        "worst_21_session": float(rolling.min()),
        "worst_calendar_year": float(yearly.min()),
        "worst_calendar_year_label": int(yearly.idxmin()),
    }


def window_stats(net: pd.Series, scale: pd.Series, cost: pd.Series, crash_windows: Sequence[Sequence[str]]
                 ) -> dict[str, Any]:
    stats = annualized_stats(net)
    stats.update(drawdown_stats(net))
    stats["mean_scale"] = float(scale.mean())
    stats["fraction_below_full"] = float((scale < 1.0).mean())
    stats["total_cost"] = float(cost.sum())
    crashes = {}
    for start, end in crash_windows:
        piece = net.loc[start:end]
        in_range = net.index[0] <= pd.Timestamp(start) and pd.Timestamp(end) <= net.index[-1]
        crashes[f"{start}..{end}"] = float(np.expm1(np.log1p(piece).sum())) if in_range and len(piece) else None
    stats["crash_windows"] = crashes
    return stats


def interval_record(rule_net: pd.Series, none_net: pd.Series, spec: SharpeBootstrapSpec, label: str,
                    per_year: int = 252) -> dict[str, Any]:
    joined = pd.concat([rule_net, none_net], axis=1, join="inner").dropna()
    result = sharpe_difference_interval(joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy(), spec,
                                        label=label).to_dict()
    root = math.sqrt(per_year)
    result["annualized"] = {key: result[key] * root for key in ("sharpe_a", "sharpe_b", "difference", "lower",
                                                                "upper")}
    result["first"], result["last"] = str(joined.index[0].date()), str(joined.index[-1].date())
    return result


def crash_rule_verdict(interval: Mapping[str, Any], rule_window: Mapping[str, Any],
                       none_window: Mapping[str, Any]) -> str:
    """docs/public_factor_lab_design.md §5, crash_degross_v1, read on pre-2020."""
    below_zero = interval["upper"] < 0
    lower_dd = rule_window["max_drawdown"] < none_window["max_drawdown"]
    lower_worst = rule_window["worst_21_session"] > none_window["worst_21_session"]
    if below_zero or not lower_dd:
        return "contradicted"
    if lower_dd and lower_worst:
        return "supported"
    return "inconclusive"


def published_rule_verdict(full: Mapping[str, Any], post: Mapping[str, Any]) -> str:
    """docs/public_factor_lab_design.md §5, constant_vol_bsc and variance_managed_mm."""
    if full["upper"] < 0 or post["upper"] < 0:
        return "contradicted"
    if full["lower"] > 0 and post["difference"] >= 0:
        return "supported"
    return "inconclusive"


def combine_verdicts(verdicts: Sequence[str]) -> str:
    if "contradicted" in verdicts:
        return "contradicted"
    if all(verdict == "supported" for verdict in verdicts):
        return "supported"
    return "inconclusive"


# --------------------------------------------------------------------- sizing driver


def arm_config(reg: Mapping[str, Any], base: str, rule: str) -> dict[str, Any]:
    sizing = reg["sizing"]
    config = {
        "registration": reg["id"],
        "selection_set": reg["selection_set"],
        "base": base,
        "rule": rule,
        "refresh_sessions": sizing["refresh_sessions"],
        "warmup_sessions": sizing["warmup_sessions"],
        "max_scale": sizing["max_scale"],
        "base_gross": sizing["base_gross"],
        "scale_change_cost_bps": sizing["scale_change_cost_bps"],
        "base_cost_per_gross_year": sizing["base_cost_per_gross_year"][base],
    }
    if rule in sizing:
        config["rule_params"] = dict(sizing[rule])
    return config


def base_calendar(reg: Mapping[str, Any], data: PublicData, base: str) -> tuple[pd.Series, list[str]]:
    """The base series on the registered calendar, and the sessions dropped to get there."""
    if reg["sizing"]["calendar"] != "intersect_market":
        raise LabError(f"unknown calendar rule {reg['sizing']['calendar']!r}")
    returns = data.series[base]
    keep = returns.index.isin(data.market_total.index)
    dropped = [str(day.date()) for day in returns.index[~keep]]
    return returns[keep], dropped


def compute_arm(reg: Mapping[str, Any], data: PublicData, base: str, rule: str) -> pd.DataFrame:
    sizing = reg["sizing"]
    returns = base_calendar(reg, data, base)[0]
    kwargs: dict[str, Any] = {"refresh": int(sizing["refresh_sessions"]), "max_scale": float(sizing["max_scale"])}
    if rule == "crash_degross_v1":
        params = sizing[rule]
        market = data.market_total.reindex(returns.index)
        if market.isna().any():
            raise LabError(f"market series misses {int(market.isna().sum())} sessions of {base}")
        kwargs.update(market_total=market.to_numpy(), theta=float(params["theta"]),
                      trailing_sessions=int(params["trailing_sessions"]), multiplier=float(params["multiplier"]),
                      exit_clean_sessions=int(params["exit_clean_sessions"]))
    elif rule == "constant_vol_bsc":
        params = sizing[rule]
        kwargs.update(target_vol=float(params["target_vol"]), lookback_sessions=int(params["lookback_sessions"]))
    elif rule == "variance_managed_mm":
        params = sizing[rule]
        kwargs.update(lookback_sessions=int(params["lookback_sessions"]),
                      min_history_refreshes=int(params["min_history_refreshes"]))
    values = returns.to_numpy(dtype=float)
    scale = RULES[rule](values, **kwargs)
    net, cost = apply_scale(values, scale, base_gross=float(sizing["base_gross"]),
                            change_cost_bps=float(sizing["scale_change_cost_bps"]),
                            base_cost_per_gross_year=float(sizing["base_cost_per_gross_year"][base]))
    frame = pd.DataFrame({"net": net, "scale": scale, "cost": cost, "base": values}, index=returns.index)
    return frame.iloc[int(sizing["warmup_sessions"]):]


def _prior_rows(log: TrialLog, selection_set: str) -> list[dict[str, Any]]:
    return [row for row in log.verify() if row.get("selection_set_id") == selection_set]


def _arm_status(rows: Sequence[Mapping[str, Any]], arm_id: str) -> str | None:
    """Last terminal event recorded for ``arm_id``, or ``started`` if it never ended."""
    runs = [row for row in rows if row.get("arm_id") == arm_id]
    if not runs:
        return None
    terminal = [row["event"] for row in runs if row["event"] != STARTED]
    return terminal[-1] if terminal else STARTED


def run_sizing(reg: Mapping[str, Any], data: PublicData, fetched: Mapping[str, Mapping[str, Any]], out: Path,
               log_path: Path, kind: str, code_commit: str) -> dict[str, Any]:
    sizing = reg["sizing"]
    log = TrialLog(log_path)
    existing = _prior_rows(log, reg["selection_set"]) if log.exists() else []
    arms = [(base, rule) for base in sizing["bases"] for rule in sizing["rules"]]
    if len(arms) != int(reg["trial_budget"]):
        raise LabError(f"{len(arms)} declared arms but trial_budget {reg['trial_budget']}")
    sources_for = {"UMD": ["ff_mom_daily", "ff3_daily"], "RESMOM": ["jkp_resmom_daily", "ff3_daily"]}
    frames: dict[str, pd.DataFrame] = {}
    for base, rule in arms:
        arm_id = f"{base}:{rule}"
        status = _arm_status(existing, arm_id)
        if kind == "new" and status is not None:
            raise LabError(f"arm {arm_id} already has a run ({status}); a second evaluation is not a new trial")
        if kind == "retry":
            if status is None:
                raise LabError(f"arm {arm_id} has no first attempt to retry")
            if status == FINISHED:
                continue  # read back below by deterministic recomputation, not re-counted
        config = arm_config(reg, base, rule)
        config_id = hashlib.sha256(stable_json_dumps(config).encode()).hexdigest()
        data_id = hashlib.sha256(
            "|".join(fetched[source]["sha256"] for source in sources_for[base]).encode()
        ).hexdigest()
        recorder = RunRecorder(log=log, run_id=f"run-{uuid.uuid4().hex[:16]}", config_id=config_id, data_id=data_id,
                               kind=kind, selection_set_id=reg["selection_set"], arm_id=arm_id)
        recorder.started()
        try:
            frame = compute_arm(reg, data, base, rule)
            sharpe_p = periodic_sharpe(frame["net"].to_numpy())
            recorder.finished({
                "periodic_sharpe": sharpe_p,
                "annualized_sharpe": sharpe_p * math.sqrt(252),
                "n_sessions": int(len(frame)),
                "first": str(frame.index[0].date()),
                "last": str(frame.index[-1].date()),
                "code_commit": code_commit,
            })
        except Exception as exc:
            recorder.failed(f"{type(exc).__name__}: {exc}")
            raise
        frames[arm_id] = frame
    missing = [f"{b}:{r}" for b, r in arms if f"{b}:{r}" not in frames]
    for arm_id in missing:  # finished on an earlier attempt: recompute deterministically for the summary
        base, rule = arm_id.split(":")
        frames[arm_id] = compute_arm(reg, data, base, rule)
    dropped = {base: base_calendar(reg, data, base)[1] for base in sizing["bases"]}
    return summarize_sizing(reg, frames, out, calendar_drops=dropped)


def summarize_sizing(reg: Mapping[str, Any], frames: Mapping[str, pd.DataFrame], out: Path,
                     calendar_drops: Mapping[str, list[str]] | None = None) -> dict[str, Any]:
    sizing = reg["sizing"]
    boot = sizing["bootstrap"]
    spec = SharpeBootstrapSpec(block_bars=int(boot["block_bars"]), n_resamples=int(boot["n_resamples"]),
                               seed=int(boot["seed"]), confidence=float(boot["confidence"]),
                               decision_interval_bars=int(sizing["refresh_sessions"]))
    trial_sharpes = [periodic_sharpe(frame["net"].to_numpy()) for frame in frames.values()]
    pre_end = sizing["pre_2020_end"]
    windows_for = {
        "none": ("full", "pre_2020", "since_2020"),
        "crash_degross_v1": ("full", "pre_2020", "since_2020"),
        "constant_vol_bsc": ("full", "pre_2020", "since_2020", "post_publication"),
        "variance_managed_mm": ("full", "pre_2020", "since_2020", "post_publication"),
    }

    def cut(frame: pd.DataFrame, rule: str, window: str) -> pd.DataFrame:
        if window == "full":
            return frame
        if window == "pre_2020":
            return frame.loc[:pre_end]
        if window == "since_2020":
            return frame.loc["2020-01-01":]
        if window == "post_publication":
            return frame.loc[sizing[rule]["publication"] + "-01":]
        raise LabError(f"unknown window {window!r}")

    arms: dict[str, Any] = {}
    for arm_id, frame in frames.items():
        base, rule = arm_id.split(":")
        record: dict[str, Any] = {
            "base": base,
            "rule": rule,
            "dsr_vs_set": deflated_sharpe_ratio(frame["net"].to_numpy(), trial_sharpes),
            "windows": {},
            "intervals_vs_none": {},
        }
        for window in windows_for[rule]:
            piece = cut(frame, rule, window)
            record["windows"][window] = window_stats(piece["net"], piece["scale"], piece["cost"],
                                                     sizing["crash_windows"])
            if rule != "none":
                control = cut(frames[f"{base}:none"], rule, window)
                record["intervals_vs_none"][window] = interval_record(piece["net"], control["net"], spec,
                                                                      f"{arm_id}-minus-none:{window}")
        arms[arm_id] = record

    verdicts: dict[str, Any] = {}
    for rule in ("crash_degross_v1", "constant_vol_bsc", "variance_managed_mm"):
        per_base = {}
        for base in sizing["bases"]:
            record = arms[f"{base}:{rule}"]
            if rule == "crash_degross_v1":
                per_base[base] = crash_rule_verdict(record["intervals_vs_none"]["pre_2020"],
                                                    record["windows"]["pre_2020"],
                                                    arms[f"{base}:none"]["windows"]["pre_2020"])
            else:
                per_base[base] = published_rule_verdict(record["intervals_vs_none"]["full"],
                                                        record["intervals_vs_none"]["post_publication"])
        verdicts[rule] = {"per_base": per_base, "overall": combine_verdicts(list(per_base.values()))}

    summary = {"registration": reg["id"], "selection_set": reg["selection_set"], "bootstrap": spec.method(),
               "calendar_drops": dict(calendar_drops or {}), "arms": arms, "verdicts": verdicts}
    (out / "sizing").mkdir(parents=True, exist_ok=True)
    (out / "sizing" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    series = pd.concat({arm_id: frame[["net", "scale", "cost"]] for arm_id, frame in frames.items()}, axis=1)
    series.columns = [f"{arm}|{field}" for arm, field in series.columns]
    series.to_parquet(out / "sizing" / "returns.parquet")
    return summary


# --------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public factor lab (registration 2026-09-factor-lab).")
    parser.add_argument("--part", choices=("fetch", "priors", "fidelity", "sizing", "all"), default="all")
    parser.add_argument("--kind", choices=("new", "retry"), default="new")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args(argv)

    reg = load_registration()
    fetched = fetch_sources(reg["sources"], CACHE_DIR, refresh=args.refresh_cache)
    if args.part == "fetch":
        print(json.dumps(fetched, indent=2))
        return 0
    code_commit = require_committed([REGISTRATION, DESIGN_DOC])
    data = build_public_data(reg, fetched)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sources.json").write_text(json.dumps(
        {"sources": fetched, "vintages": data.vintages, "code_commit": code_commit}, indent=2, sort_keys=True))
    priors_result = None
    if args.part in ("priors", "fidelity", "all"):
        priors_result = run_priors(reg, data, OUT_DIR)
    if args.part in ("fidelity", "all"):
        assert priors_result is not None
        run_fidelity(reg, data, priors_result["priors"], OUT_DIR)
    if args.part in ("sizing", "all"):
        summary = run_sizing(reg, data, fetched, OUT_DIR, TRIAL_LOG, args.kind, code_commit)
        print(json.dumps(summary["verdicts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
