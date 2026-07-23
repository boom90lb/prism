"""Tax-wedge measurement over the recorded ledgers (docs/tax_wedge_spec.md, ratified 2026-07-19).

The constitutional viability gate stays pre-tax; this instrument computes the
spec's four *asymmetries* against a symmetric contemporaneous-τ baseline
(under which excess-over-cash and its volatility both scale by (1−τ) and the
periodic Sharpe comparison is τ-invariant, spec §2):

1. **Wash-sale deferral** — replayed over the actual lot sequence of the
   replay fills ledger (``runs/replay_floor_1000000/fills.jsonl``), FIFO
   lots, the 30-day window on both sides, disallowed losses attached to the
   replacement lots' basis. No model. The certified run directory
   (``results/demotion_b1``) records weight-space accounting only — no
   per-lot fills — so it CANNOT support this asymmetry; the ledger
   substitution is recorded in the JSON meta and ``docs/tax_wedge.md``,
   never silent.
2. **Payments-in-lieu on the short leg** — the dividend-wedge data path
   (``research/scripts/dividend_wedge.py`` accrual over the certified
   ``target_weights.csv`` and the cached Alpaca corporate-action records;
   no network I/O — the records cache is read, never refetched). Treatment:
   short PIL is capitalized into the short's capital result (IRC §263(h),
   sub-46-day shorts — the monthly cadence holds shorts ~30 calendar days),
   so in net-capital-gain years relief matches the baseline and in the loss
   year the PIL deepens the capped loss; the no-relief (−τ·PIL) and
   full-relief (0) bounds are reported beside the measured number. The long
   leg's dividends are ordinary income at τ — the 61-day qualified holding
   window fails at monthly cadence — which is exactly the baseline's
   treatment, so the qualified-side wedge vs the symmetric-τ baseline is an
   explicit, derived zero (stated, not silent); the foregone qualified-rate
   benefit needs a qualified-rate parameter and lot holding periods outside
   the ratified parameter set (spec §4) and is recorded, not answered.
3. **Capital-loss netting cap** — gains taxed at τ in the year earned; a net
   loss year deducts at most ``--loss-cap`` against ordinary income and
   carries the remainder forward against later gains. The wedge is the
   terminal unrelieved carryforward plus the time value of tax prepaid
   relative to the baseline, monetized at the run's own recorded T-bill
   hurdle (``summary.json`` ``after_cost_hurdle``) — no new rate constants.
4. **State asymmetry** — the T-bill hurdle's interest is state-exempt while
   trading gains are not: −τ_state × hurdle per year, scaling with the state
   rate parameter.

Sign convention: negative bps/yr = after-tax drag vs the symmetric-τ
baseline (same orientation as the flatten −19.4 and dividend −36.1 wedges).
Both spec §3 outputs are mandatory and emitted: the steady-state annualized
wedge over the certified window, and the crash-year conditional wedge on (a)
the worst calendar year in the sample and (b) a synthetic −30% book-return
year (the −30% is the spec §3 pinned constant). Federal rate, state rate,
loss cap and book equity are CLI parameters with no defaults (spec §4 — a
committed artifact records the vector it was computed under; the tracked
reference run uses a labeled REFERENCE vector, not the operator's rates).

Uncounted diagnostic: searches nothing, changes no trial code path, appends
nothing to the trials ledger, moves no ratified statistic, writes one JSON.
Frame and results: ``docs/tax_wedge.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from research.scripts.dividend_wedge import _load_closes, dividend_panel

# ---------------------------------------------------------------------------
# asymmetry 1 — wash-sale deferral over real lots (pure, unit-tested)
# ---------------------------------------------------------------------------


def fifo_wash_replay(fills: list[dict], wash_window_days: int = 30) -> dict:
    """Replay the fills ledger through FIFO lots with the wash-sale rule.

    ``fills`` rows need ``symbol``, ``filled_bar`` (ISO date), ``qty``
    (signed shares) and ``fill_price``. A realized loss on a long close or a
    short cover is disallowed to the extent same-direction replacement
    shares were opened within ``wash_window_days`` before or after the
    closing date (the closed lot's own opening never replaces itself); the
    disallowed amount attaches to the replacement lot's basis and surfaces
    when that lot closes, so re-washed chains compound naturally. Wash
    matching is per symbol — no substantially-identical matching across
    tickers. Conservation invariant (asserted): naive − taxable realized
    P&L summed over the sample equals the deferral still attached to lots
    open at the ledger end.
    """
    by_symbol: dict[str, list[dict]] = {}
    for i, f in enumerate(fills):
        row = {
            "seq": i,
            "date": date.fromisoformat(str(f["filled_bar"])[:10]),
            "qty": float(f["qty"]),
            "price": float(f["fill_price"]),
        }
        by_symbol.setdefault(str(f["symbol"]), []).append(row)

    window = timedelta(days=wash_window_days)
    closes: list[dict] = []  # realized close events, all symbols
    opens: dict[str, list[dict]] = {}  # symbol -> opening events (capacity for replacement designation)
    lots_by_symbol: dict[str, dict[str, list[dict]]] = {}

    # Pass 1: share mechanics (basis-independent) — decompose every fill into
    # lot closes and opens, FIFO within direction.
    for sym, rows in by_symbol.items():
        rows.sort(key=lambda r: (r["date"], r["seq"]))
        long_lots: list[dict] = []
        short_lots: list[dict] = []
        sym_opens: list[dict] = []
        for r in rows:
            remaining = abs(r["qty"])
            direction = "long" if r["qty"] > 0 else "short"
            against = short_lots if direction == "long" else long_lots
            while remaining > 0 and against:
                lot = against[0]
                take = min(remaining, lot["shares"])
                closes.append(
                    {
                        "symbol": sym,
                        "date": r["date"],
                        "direction": lot["direction"],
                        "shares": take,
                        "open_price": lot["price"],
                        "close_price": r["price"],
                        "lot_id": lot["lot_id"],
                        "close_seq": r["seq"],
                    }
                )
                lot["shares"] -= take
                remaining -= take
                if lot["shares"] == 0:
                    against.pop(0)
            if remaining > 0:
                lot = {
                    "lot_id": r["seq"],
                    "direction": direction,
                    "shares": remaining,
                    "price": r["price"],
                    "date": r["date"],
                }
                (long_lots if direction == "long" else short_lots).append(lot)
                sym_opens.append(
                    {"lot_id": r["seq"], "direction": direction, "date": r["date"], "capacity": remaining}
                )
        opens[sym] = sym_opens
        lots_by_symbol[sym] = {"long": long_lots, "short": short_lots}

    # Pass 2: chronological realization with wash disallowance and deferral.
    closes.sort(key=lambda c: (c["date"], c["close_seq"]))
    deferred: dict[tuple[str, int], float] = {}  # (symbol, lot_id) -> attached $ not yet recognized
    lot_open_shares: dict[tuple[str, int], float] = {
        (sym, o["lot_id"]): o["capacity"] for sym, sym_opens in opens.items() for o in sym_opens
    }
    naive_by_year: dict[int, float] = {}
    taxable_by_year: dict[int, float] = {}
    n_loss_events = 0
    n_wash_events = 0
    n_replacement_lots_already_closed = 0
    total_disallowed = 0.0
    total_recognized_deferral = 0.0

    for c in closes:
        key = (c["symbol"], c["lot_id"])
        raw = (
            (c["close_price"] - c["open_price"]) * c["shares"]
            if c["direction"] == "long"
            else (c["open_price"] - c["close_price"]) * c["shares"]
        )
        open_left = lot_open_shares[key]
        attach_pool = deferred.get(key, 0.0)
        attached = attach_pool * (c["shares"] / open_left) if open_left > 0 else 0.0
        deferred[key] = attach_pool - attached
        lot_open_shares[key] = open_left - c["shares"]
        total_recognized_deferral += attached

        recognized = raw - attached
        disallowed = 0.0
        if recognized < 0.0:
            n_loss_events += 1
            loss_per_share = -recognized / c["shares"]
            need = c["shares"]
            for o in opens[c["symbol"]]:
                if need <= 0:
                    break
                if o["direction"] != c["direction"] or o["lot_id"] == c["lot_id"] or o["capacity"] <= 0:
                    continue
                if abs((o["date"] - c["date"]).days) > window.days:
                    continue
                if o["date"] <= c["date"] and lot_open_shares[(c["symbol"], o["lot_id"])] <= 0:
                    # A backward-window replacement lot already fully closed:
                    # nothing left to carry the deferral. Counted, not silent.
                    n_replacement_lots_already_closed += 1
                    continue
                take = min(need, o["capacity"])
                o["capacity"] -= take
                need -= take
                amount = loss_per_share * take
                disallowed += amount
                rkey = (c["symbol"], o["lot_id"])
                deferred[rkey] = deferred.get(rkey, 0.0) + amount
            if disallowed > 0.0:
                n_wash_events += 1
                total_disallowed += disallowed

        year = c["date"].year
        naive_by_year[year] = naive_by_year.get(year, 0.0) + raw
        taxable_by_year[year] = taxable_by_year.get(year, 0.0) + recognized + disallowed

    deferred_open_at_end = total_disallowed - total_recognized_deferral
    conservation = sum(naive_by_year.values()) - sum(taxable_by_year.values())
    assert abs(conservation + deferred_open_at_end) < 1e-6, (
        f"wash replay conservation broken: naive-taxable {conservation:.6f} vs "
        f"-deferred_open {-deferred_open_at_end:.6f}"
    )
    return {
        "naive_realized_by_year": naive_by_year,
        "taxable_realized_by_year": taxable_by_year,
        "n_close_events": len(closes),
        "n_loss_events": n_loss_events,
        "n_wash_events": n_wash_events,
        "n_replacement_lots_already_closed": n_replacement_lots_already_closed,
        "total_disallowed": total_disallowed,
        "recognized_deferral_in_sample": total_recognized_deferral,
        "deferred_open_at_end": deferred_open_at_end,
    }


def wash_wedge(replay: dict, tau: float, hurdle_rate: float, avg_equity: float, years: float) -> dict:
    """Monetize the wash deferral: tax prepaid on the sample-end deferral balance.

    Deferral is a timing drag, not loss destruction (spec §2.1): the cost is
    tax paid earlier than the baseline would have collected it. The
    sample-end deferred balance is monetized as prepaid tax carried at the
    run's recorded T-bill hurdle; the one-time prepaid amount is reported
    beside the annualized carrying cost.
    """
    prepaid_tax = tau * replay["deferred_open_at_end"]
    carry_cost_per_year = prepaid_tax * hurdle_rate
    return {
        "prepaid_tax_usd": prepaid_tax,
        "prepaid_one_time_bps": -prepaid_tax / avg_equity * 1e4,
        "carry_cost_usd_per_year": carry_cost_per_year,
        "wedge_bps_per_year": -carry_cost_per_year / avg_equity * 1e4,
        "ledger_years": years,
    }


# ---------------------------------------------------------------------------
# asymmetry 3 (and the engine asymmetry 2 rides through) — loss-cap netting
# ---------------------------------------------------------------------------


def capital_engine(pnl_by_year: dict[int, float], loss_cap: float, tau: float, hurdle_rate: float) -> dict:
    """Year-by-year after-tax delta vs the symmetric-τ baseline under the loss cap.

    Baseline: every year's net P&L taxed (or refunded) at τ contemporaneously.
    Actual: gains taxed at τ; a net-loss year deducts at most ``loss_cap``
    against ordinary income and carries the remainder forward; carryforward
    nets against later gains first, then feeds the ordinary cap. Timing cost
    charges the prepaid-tax balance (τ × carryforward) at ``hurdle_rate`` for
    each year-boundary it persists; the terminal component is the tax value
    of carryforward still unrelieved at the sample end. Undiscounted in-sample
    reversals therefore net to zero by construction — the wedge is exactly
    timing plus terminal, which is what the asymmetry is.
    """
    rows = []
    carryforward = 0.0
    timing_cost = 0.0
    delta_total = 0.0
    years = sorted(pnl_by_year)
    for i, year in enumerate(years):
        net = float(pnl_by_year[year])
        baseline_tax = tau * net
        if net >= 0.0:
            used_cf = min(carryforward, net)
            carryforward -= used_cf
            taxable_gain = net - used_cf
            ordinary_offset = min(loss_cap, carryforward)
            carryforward -= ordinary_offset
            tax = tau * taxable_gain - tau * ordinary_offset
        else:
            pool = carryforward - net
            ordinary_offset = min(loss_cap, pool)
            carryforward = pool - ordinary_offset
            tax = -tau * ordinary_offset
        delta = baseline_tax - tax  # after-tax income vs baseline; negative = drag
        delta_total += delta
        timing_year = tau * carryforward * hurdle_rate if i < len(years) - 1 else 0.0
        timing_cost += timing_year
        rows.append(
            {
                "year": year,
                "net_pnl": net,
                "baseline_tax": baseline_tax,
                "actual_tax": tax,
                "after_tax_delta": delta,
                "carryforward_end": carryforward,
                "timing_cost": timing_year,
            }
        )
    terminal = tau * carryforward
    assert abs(delta_total + terminal) < 1e-6, (
        f"capital engine conservation broken: sum of deltas {delta_total:.6f} vs -terminal {-terminal:.6f}"
    )
    return {
        "per_year": rows,
        "carryforward_end": carryforward,
        "terminal_unrelieved_tax": terminal,
        "timing_cost_total": timing_cost,
        "after_tax_delta_total": delta_total,
    }


def loss_cap_wedge(engine: dict, equity: float, years: float) -> dict:
    """Engine output → annualized bps on ``equity`` (negative = drag)."""
    total_cost = engine["terminal_unrelieved_tax"] + engine["timing_cost_total"]
    return {
        "terminal_unrelieved_tax_usd": engine["terminal_unrelieved_tax"],
        "timing_cost_usd": engine["timing_cost_total"],
        "wedge_bps_per_year": -total_cost / equity / years * 1e4,
    }


# ---------------------------------------------------------------------------
# asymmetry 2 — payments-in-lieu vs the long leg's dividend treatment
# ---------------------------------------------------------------------------


def pil_wedge(
    price_pnl_by_year: dict[int, float],
    pil_paid_by_year: dict[int, float],
    long_dividends_by_year: dict[int, float],
    loss_cap: float,
    tau: float,
    hurdle_rate: float,
    equity: float,
    years: float,
) -> dict:
    """PIL capitalization wedge (measured) with its no-relief / full-relief bounds.

    The measured number is the attribution difference between the capital
    engine run on (price P&L − PIL) and on price P&L alone: capitalized PIL
    (§263(h)) is relieved at τ in net-gain years exactly as the baseline
    relieves it, and deepens the capped loss in a net-loss year — the
    interaction the spec flags as "worsens after tax". The long leg's
    dividends are ordinary at τ (61-day qualified window fails at monthly
    cadence), identical to the baseline: a derived zero, stated loudly.
    """
    with_pil = capital_engine(
        {y: price_pnl_by_year.get(y, 0.0) - pil_paid_by_year.get(y, 0.0) for y in price_pnl_by_year},
        loss_cap,
        tau,
        hurdle_rate,
    )
    without_pil = capital_engine(price_pnl_by_year, loss_cap, tau, hurdle_rate)
    cost_with = with_pil["terminal_unrelieved_tax"] + with_pil["timing_cost_total"]
    cost_without = without_pil["terminal_unrelieved_tax"] + without_pil["timing_cost_total"]
    pil_annual = sum(pil_paid_by_year.values()) / years
    long_annual = sum(long_dividends_by_year.values()) / years
    return {
        "treatment": (
            "short PIL capitalized into the short's capital result (IRC 263(h), sub-46-day shorts at "
            "monthly cadence); long dividends ordinary at tau (61-day qualified window fails), which "
            "matches the baseline exactly"
        ),
        "wedge_bps_per_year": -(cost_with - cost_without) / equity / years * 1e4,
        "bound_no_relief_bps_per_year": -tau * pil_annual / equity * 1e4,
        "bound_full_relief_bps_per_year": 0.0,
        "long_leg_qualified_wedge_bps_per_year": 0.0,
        "long_leg_note": (
            "zero vs the symmetric-tau baseline BY DERIVATION, not by omission: both tax long dividends "
            "as ordinary income at tau; the foregone qualified-rate benefit relative to a "
            "qualified-eligible holder needs a qualified-rate parameter and lot holding periods outside "
            "the ratified parameter set (spec section 4) - recorded, not answered"
        ),
        "pil_paid_annualized_usd": pil_annual,
        "long_dividends_annualized_usd": long_annual,
        "engine_with_pil": with_pil,
        "engine_without_pil": without_pil,
    }


# ---------------------------------------------------------------------------
# asymmetry 4 — state exemption on the T-bill hurdle
# ---------------------------------------------------------------------------


def state_exemption_wedge(hurdle_annual_pct: float, state_rate: float) -> dict:
    """T-bill interest is state-exempt; trading gains are not: −τ_state × hurdle."""
    return {
        "hurdle_annual_pct": hurdle_annual_pct,
        "wedge_bps_per_year": -state_rate * hurdle_annual_pct * 100.0,
    }


# ---------------------------------------------------------------------------
# spec §3 crash-year conditional cells
# ---------------------------------------------------------------------------


def worst_year_cell(engine_with_pil: dict, state_bps: float, equity: float) -> dict:
    """Conditional wedge on the sample's worst calendar year (spec §3, output 2a).

    The cell reports the crash year's own after-tax gap vs the baseline —
    the refund the baseline pays that year and the capped relief the actual
    code allows — BEFORE any later-year recovery, which is exactly the
    convexity penalty the sizing read consumes; whether the carryforward was
    later relieved in sample is reported beside it, not averaged into it.
    """
    rows = engine_with_pil["per_year"]
    worst = min(rows, key=lambda r: r["net_pnl"])
    later = [r for r in rows if r["year"] > worst["year"]]
    recovered = bool(later) and later[-1]["carryforward_end"] == 0.0
    capital_bps = worst["after_tax_delta"] / equity * 1e4
    return {
        "year": worst["year"],
        "net_pnl_usd": worst["net_pnl"],
        "capital_cell_bps": capital_bps,
        "state_bps": state_bps,
        "wash_bps": None,
        "wash_note": (
            "unsupported for this cell: the only per-lot fills ledger in-tree covers 2026 sessions, "
            "not the worst sample year - reported null, never zero (N7)"
        ),
        "conditional_wedge_bps": capital_bps + state_bps,
        "carryforward_relieved_later_in_sample": recovered,
    }


def synthetic_crash_cell(
    equity: float,
    loss_cap: float,
    tau: float,
    pil_annual_usd: float,
    state_bps: float,
    book_return: float = -0.30,
) -> dict:
    """Conditional wedge on a synthetic −30% book-return year (spec §3, output 2b).

    The −30% is the spec-pinned constant. The year's capital result is the
    synthetic book loss deepened by the steady-state annual PIL (capitalized,
    so it joins the capped loss); the baseline refunds the whole loss at τ,
    the actual code relieves ``loss_cap`` against ordinary income in-year.
    """
    net = book_return * equity - pil_annual_usd
    loss = -net
    baseline_refund = tau * loss
    actual_relief = tau * min(loss_cap, loss)
    capital_bps = -(baseline_refund - actual_relief) / equity * 1e4
    return {
        "book_return": book_return,
        "net_pnl_usd": net,
        "capital_cell_bps": capital_bps,
        "state_bps": state_bps,
        "wash_bps": None,
        "wash_note": "no lot ledger exists for a synthetic year - reported null, never zero (N7)",
        "conditional_wedge_bps": capital_bps + state_bps,
    }


# ---------------------------------------------------------------------------
# ledger loaders (local committed artifacts only — no network path at all)
# ---------------------------------------------------------------------------


def load_book_pnl_by_year(run_dir: Path, equity: float) -> tuple[dict[int, float], float, list[str]]:
    """Certified daily returns → calendar-year P&L dollars on a constant equity base.

    Constant-base convention (each year's compounded return × ``equity``):
    both the actual and baseline tax codes consume the same series, so the
    wedge is first-order insensitive to the convention; it is recorded in
    the JSON meta. Returns ``(pnl_by_year, window_years, span)``.
    """
    frame = pd.read_csv(run_dir / "returns.csv", index_col=0)
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert("America/New_York")
    frame.index = idx.tz_localize(None).normalize()
    returns = frame["daily_return"].astype(float)
    yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    span = [str(returns.index[0].date()), str(returns.index[-1].date())]
    return {int(y): float(r) * equity for y, r in yearly.items()}, years, span


def load_leg_flows_by_year(
    run_dir: Path, records_json: Path, data_dir: Path, cache_suffix: str, equity: float
) -> tuple[dict[int, float], dict[int, float], dict]:
    """Dividend-wedge accrual split by leg and calendar year, in dollars on ``equity``.

    Reuses the dividend-wedge instrument's own machinery (``dividend_panel``
    accrual ``w[t-1]·amount[t]/close[t-1]``) over the certified weights and
    the cached corporate-action records — read from disk, never refetched.
    Returns ``(pil_paid_by_year, long_dividends_by_year, meta)`` where PIL
    paid is a positive dollar amount per year.
    """
    if not records_json.exists():
        raise SystemExit(
            f"dividend records cache {records_json} missing - asymmetry 2 has no local data source and "
            "a diagnostic never fetches (N7): refusing to emit a silent-empty PIL cell"
        )
    weights = pd.read_csv(run_dir / "target_weights.csv", index_col=0)
    widx = pd.DatetimeIndex(pd.to_datetime(weights.index, utc=True)).tz_convert("America/New_York")
    weights.index = widx.tz_localize(None).normalize()
    ever_held = sorted(weights.columns[(weights != 0.0).any(axis=0)])
    weights = weights[ever_held]

    records = json.loads(records_json.read_text())
    closes = _load_closes(ever_held, data_dir, cache_suffix, weights.index)
    missing_closes = sorted(set(ever_held) - set(closes.columns[closes.notna().any(axis=0)]))
    dividends, div_meta = dividend_panel(records, weights.index, weights.columns)

    prev_w = weights.shift(1)
    flow = (prev_w * (dividends / closes.shift(1))).fillna(0.0)
    long_flow = flow.where(prev_w > 0.0, 0.0).sum(axis=1)
    short_flow = flow.where(prev_w < 0.0, 0.0).sum(axis=1)
    window_years = (weights.index[-1] - weights.index[0]).days / 365.25

    pil_by_year = {int(y): float(-v) * equity for y, v in short_flow.groupby(short_flow.index.year).sum().items()}
    long_by_year = {int(y): float(v) * equity for y, v in long_flow.groupby(long_flow.index.year).sum().items()}
    meta = {
        **div_meta,
        "n_ever_held": len(ever_held),
        "missing_close_caches": missing_closes,
        "short_flow_annualized_bps": float(short_flow.sum() / window_years * 1e4),
        "long_flow_annualized_bps": float(long_flow.sum() / window_years * 1e4),
    }
    return pil_by_year, long_by_year, meta


def load_replay_fills(replay_dir: Path) -> tuple[list[dict], float, float, list[str]]:
    """Replay fills + equity ledgers → (fills, average equity, span years, span).

    Missing ledgers are a named failure, not an empty wedge: asymmetry 1 has
    no other per-lot source in-tree.
    """
    fills_path = replay_dir / "fills.jsonl"
    equity_path = replay_dir / "equity.jsonl"
    if not fills_path.exists() or not equity_path.exists():
        raise SystemExit(
            f"replay ledgers missing under {replay_dir} - asymmetry 1 (wash-sale) has no per-lot fills "
            "source in-tree; refusing to emit a silent-empty wash cell (N7)"
        )
    fills = [json.loads(line) for line in fills_path.read_text().splitlines() if line.strip()]
    equity_rows = [json.loads(line) for line in equity_path.read_text().splitlines() if line.strip()]
    equity = pd.Series(
        {r["decision_bar"]: float(r["equity"]) for r in equity_rows}
    ).sort_index()
    first, last = date.fromisoformat(equity.index[0]), date.fromisoformat(equity.index[-1])
    years = (last - first).days / 365.25
    return fills, float(equity.mean()), years, [str(first), str(last)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--federal-rate", type=float, required=True, help="Federal marginal rate (spec §4: no default)")
    parser.add_argument("--state-rate", type=float, required=True, help="State marginal rate (spec §4: no default)")
    parser.add_argument("--loss-cap", type=float, required=True, help="Filing-year ordinary-income loss cap, USD")
    parser.add_argument(
        "--book-equity",
        type=float,
        required=True,
        help=(
            "Book equity in USD the normalized certified ledger is scaled to; the loss cap is a dollar "
            "figure, so the cap asymmetry only has meaning against an equity base (no default, spec §4 spirit)"
        ),
    )
    parser.add_argument("--run-dir", default="results/demotion_b1", help="Certified run directory")
    parser.add_argument(
        "--replay-dir",
        default="runs/replay_floor_1000000",
        help="Replay run directory holding the per-lot fills ledger for the wash-sale replay",
    )
    parser.add_argument(
        "--records-json",
        default="results/alpaca_cash_dividends_2021-03-30_2026-06-12.json",
        help="Cached corporate-action records from the dividend-wedge instrument (read, never refetched)",
    )
    parser.add_argument("--data-dir", default="data", help="Bar-cache directory")
    parser.add_argument(
        "--cache-suffix", default=None, help="Cache filename suffix (default: _1d_<start>_<end>.parquet from config)"
    )
    parser.add_argument("--wash-window", type=int, default=30, help="Wash-sale window, calendar days each side")
    parser.add_argument(
        "--dividend-wedge-json",
        default="results/dividend_wedge_2026-07-13.json",
        help="Committed dividend-wedge artifact to validate the recomputed leg flows against",
    )
    parser.add_argument("--out", default=None, help="Output JSON (default: results/tax_wedge_<today>.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tau = args.federal_rate + args.state_rate
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    hurdle = summary["after_cost_hurdle"]
    assert hurdle["basis"] == "tbill_nominal", (
        f"hurdle basis {hurdle['basis']!r} is not the T-bill the state-exemption term assumes"
    )
    hurdle_rate = float(hurdle["annual_pct"]) / 100.0
    cache_suffix = args.cache_suffix or f"_1d_{config['start_date']}_{config['end_date']}.parquet"

    # ---- certified-ledger inputs (asymmetries 2, 3, 4) ----
    pnl_by_year, window_years, span = load_book_pnl_by_year(run_dir, args.book_equity)
    pil_by_year, long_by_year, flow_meta = load_leg_flows_by_year(
        run_dir, Path(args.records_json), Path(args.data_dir), cache_suffix, args.book_equity
    )

    # Validation gate: the recomputed leg flows must reproduce the committed
    # dividend-wedge artifact — same records, same accrual, same weights.
    dividend_wedge_path = Path(args.dividend_wedge_json)
    if dividend_wedge_path.exists():
        committed = json.loads(dividend_wedge_path.read_text())["wedge"]
        for leg, key in (("short_leg", "short_flow_annualized_bps"), ("long_leg", "long_flow_annualized_bps")):
            diff = abs(flow_meta[key] - committed[leg]["flow_annualized_bps"])
            assert diff < 1.0, (
                f"validation gate FAILED: recomputed {leg} flow {flow_meta[key]:.2f} bps/yr differs from the "
                f"committed dividend-wedge artifact by {diff:.2f} bps - the reused data path does not "
                "reproduce its own instrument; no number below this line means anything"
            )
        flow_validation = {"artifact": str(dividend_wedge_path), "max_abs_diff_bps_tolerance": 1.0, "passed": True}
    else:
        flow_validation = {"artifact": str(dividend_wedge_path), "note": "committed artifact not present; gate not run"}

    # ---- asymmetry 1: wash-sale replay on the only per-lot ledger in-tree ----
    fills, replay_avg_equity, replay_years, replay_span = load_replay_fills(Path(args.replay_dir))
    replay = fifo_wash_replay(fills, wash_window_days=args.wash_window)
    wash = wash_wedge(replay, tau, hurdle_rate, replay_avg_equity, replay_years)

    # ---- asymmetries 2-4 on the certified ledger ----
    pil = pil_wedge(
        pnl_by_year, pil_by_year, long_by_year, args.loss_cap, tau, hurdle_rate, args.book_equity, window_years
    )
    engine_price_only = pil["engine_without_pil"]
    cap = loss_cap_wedge(engine_price_only, args.book_equity, window_years)
    state = state_exemption_wedge(float(hurdle["annual_pct"]), args.state_rate)

    steady_state_bps = (
        wash["wedge_bps_per_year"]
        + pil["wedge_bps_per_year"]
        + cap["wedge_bps_per_year"]
        + state["wedge_bps_per_year"]
    )

    # ---- spec §3 crash-year conditional cells ----
    worst = worst_year_cell(pil["engine_with_pil"], state["wedge_bps_per_year"], args.book_equity)
    synthetic = synthetic_crash_cell(
        args.book_equity,
        args.loss_cap,
        tau,
        pil["pil_paid_annualized_usd"],
        state["wedge_bps_per_year"],
    )

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "script": "research/scripts/tax_wedge.py",
            "spec": "docs/tax_wedge_spec.md (ratified 2026-07-19)",
            "status": (
                "uncounted diagnostic: searches nothing, changes no trial code path, appends "
                "nothing to the trials ledger, moves no ratified statistic"
            ),
            "parameters": {
                "federal_rate": args.federal_rate,
                "state_rate": args.state_rate,
                "combined_tau": tau,
                "loss_cap_usd": args.loss_cap,
                "book_equity_usd": args.book_equity,
                "wash_window_days": args.wash_window,
                "note": (
                    "REFERENCE parameter vector for the record's reproducibility, not the operator's "
                    "rates - the certified numbers stay pre-tax and this overlay is re-runnable under "
                    "any vector (spec sections 1 and 4)"
                ),
            },
            "sign_convention": "negative bps/yr = after-tax drag vs the symmetric-tau baseline",
            "hurdle": hurdle,
            "pnl_convention": (
                "per-year P&L = book_equity x calendar-year compounded return of the normalized certified "
                "ledger (constant-base); actual and baseline codes consume the same series, so the wedge "
                "is first-order insensitive to the convention"
            ),
            "ledger_map": {
                "wash_sale_deferral": {
                    "ledger": str(Path(args.replay_dir) / "fills.jsonl"),
                    "span": replay_span,
                    "note": (
                        "the certified run directory records weight-space accounting only - no per-lot "
                        "fills - so it CANNOT support the wash-sale replay; measured instead on the "
                        "112-session replay fills ledger (modeled fills through live-loop mechanics), "
                        "the only per-lot ledger in-tree. Recorded here, not silently substituted."
                    ),
                },
                "payments_in_lieu": {
                    "ledger": str(run_dir / "target_weights.csv"),
                    "records": args.records_json,
                    "span": span,
                },
                "loss_cap": {"ledger": str(run_dir / "returns.csv"), "span": span},
                "state_exemption": {"ledger": str(run_dir / "summary.json"), "field": "after_cost_hurdle"},
            },
            "certified_ledger_cannot_support": (
                "per-lot wash-sale replay (no fills ledger) and lot holding periods for the qualified-"
                "dividend test; both recorded above and in docs/tax_wedge.md"
            ),
            "entity_and_elections": (
                "trader-status mark-to-market under 475(f) would change the wash-sale treatment and a "
                "monthly-cadence book very likely does not qualify; CPA territory - recorded, not "
                "answered (spec section 4)"
            ),
        },
        "book": {"run_dir": str(run_dir), "span": span, "window_years": window_years, "pnl_by_year_usd": pnl_by_year},
        "dividend_flows": {**flow_meta, "validation_vs_committed_artifact": flow_validation},
        "asymmetries": {
            "wash_sale_deferral": {"replay": replay, "wedge": wash, "avg_equity_usd": replay_avg_equity},
            "payments_in_lieu": {k: v for k, v in pil.items() if not k.startswith("engine_")},
            "loss_cap": {**cap, "per_year": engine_price_only["per_year"]},
            "state_exemption": state,
        },
        "outputs": {
            "steady_state_wedge_bps_per_year": steady_state_bps,
            "steady_state_components_bps": {
                "wash_sale_deferral": wash["wedge_bps_per_year"],
                "payments_in_lieu": pil["wedge_bps_per_year"],
                "loss_cap": cap["wedge_bps_per_year"],
                "state_exemption": state["wedge_bps_per_year"],
            },
            "crash_year_conditional": {
                "worst_sample_year": worst,
                "synthetic_minus_30pct_year": synthetic,
                "constant_note": "the -30% book-return year is the spec section 3 pinned constant",
            },
        },
    }
    out_path = Path(args.out) if args.out else Path("results") / f"tax_wedge_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
