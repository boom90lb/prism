"""Historical replay of the live daily cycle (SPEC.md §7.7) — a diagnostic
instrument, not an evidence source.

Drives the *real* ``run_daily_cycle`` — write-ahead protocol, tolerant settle,
cadence gate, online band, monitor — over local historical bars with a
simulated venue, so many cycles run in minutes instead of one per trading day.
Promoted from the scratch harnesses that validated the live spine and the
momentum-book cutover before their first live cycles (2026-07-08).

What a replay is for: mechanics validation and pre-flight — does the spine run
clean and causally over N cycles, does a cutover state flatten and rebalance,
does the cadence hold, does the monitor accrue. What a replay is **not**
(machinery ≠ claim):

* **Never I-9 cost calibration.** :class:`ReplayBroker` fills at the next
  bar's open print exactly — zero spread, zero impact — because any richer
  fill needs a fill *model*, and calibrating the spread schedule against
  one's own fill model is the cost-calibration circularity docs/handoff.md
  names. The ``fills.jsonl`` a replay writes is a replay artifact; it must
  never feed ``prism.execution.spread`` calibration.
* **Never the promotion gate's live-monitor conjunct.** The concordance read
  (docs/momentum_design.md §3) is defined on the *live paper stream*; a
  replayed ``equity.jsonl`` is a modeled-fill diagnostic stream. The two must
  stay in separate run-dirs — :func:`replay_daily_cycles` refuses to write
  into a run-dir that already holds loop state, so a replay cannot land in
  (or resume) a live ledger by accident.

**upgrade-diff** lives here too, and is the one thing in this module that is
not a replay: :func:`upgrade_diff` compares what *two engine versions* decided
over a set of recorded sessions, so an upgrade can be reviewed before it is
adopted rather than by keeping a fork alive beside the live one. It compares
decisions — the book each version constructed and the orders it would submit —
and never outcomes, because fill prices and realised equity belong to the venue
and the day, not to the engine version. A session neither side could be run
over is reported as *not compared*, with the reason, and never folded into "no
differences found".

Production-import-path safe (N8): numpy/pandas only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.live.broker import Broker, DuplicateOrder, Fill, Order
from prism.live.daily import DailyBookConfig, DailyCycleResult, run_daily_cycle
from prism.live.loop import LiveLoopContext, read_targets_ledger
from prism.live.state import StateStore
from prism.signal.base import Signal

logger = logging.getLogger(__name__)


class ReplayBroker(Broker):
    """In-memory venue: each pending order fills once, at the armed next open.

    The driver arms each cycle's fill prices with :meth:`set_fill` *before*
    running the cycle, so orders decided at close *t−1* settle at bar *t*'s
    open — the N2 next-open semantics the live OPG orders carry. A name whose
    open is missing/non-finite falls back to the order's decision-close
    ``reference_price`` with a loud warning (a replay data gap, not a venue
    outcome). Positions and cash are broker truth for the loop's reconcile.
    """

    #: A simulated venue: orders decided here reach no market, so the live
    #: submission announcement must not claim this run traded (§7.7).
    submits_to_venue: bool = False

    def __init__(self, cash: float, positions: dict[str, float] | None = None) -> None:
        self._cash = float(cash)
        self._pos: dict[str, float] = dict(positions or {})
        self._orders: dict[str, Order] = {}
        self._filled: set[str] = set()
        self._open_row: pd.Series | None = None
        self._fill_bar: str | None = None

    def set_fill(self, open_row: pd.Series, bar: str) -> None:
        """Arm the open prices (and bar label) the next settle fills at."""
        self._open_row = open_row
        self._fill_bar = bar

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self._pos.items() if q != 0.0}

    def cash(self) -> float:
        return self._cash

    def submit(self, order: Order) -> None:
        if order.client_order_id in self._orders:
            raise DuplicateOrder(order.client_order_id)
        self._orders[order.client_order_id] = order

    def submitted_order_ids(self) -> set[str]:
        return set(self._orders)

    def open_order_ids(self, client_order_ids: set[str]) -> set[str]:
        """Which of these ids are still working here: submitted and not yet filled.

        A replay venue does hold orders open — between the decision at close *t*
        and the fill at *t+1*'s armed open — so answering the question is honest
        rather than convenient. It is also what lets restart recovery
        (``loop.read_order_fates``) state every order's fate on this venue
        instead of declaring them all unknown for want of a read.
        """
        return {oid for oid in client_order_ids if oid in self._orders and oid not in self._filled}

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        if self._open_row is None or self._fill_bar is None:
            raise RuntimeError("ReplayBroker has no armed fill bar — call set_fill first (N7)")
        fills: list[Fill] = []
        for oid in sorted(client_order_ids & set(self._orders)):
            if oid in self._filled:
                continue
            order = self._orders[oid]
            raw = self._open_row.get(order.symbol)
            price = float(raw) if raw is not None and np.isfinite(raw) and raw > 0 else None
            if price is None:
                logger.warning(
                    "no open print for %s at %s — filling at the decision-close reference "
                    "%.4f (replay data gap; a live venue would have expired the order)",
                    order.symbol,
                    self._fill_bar,
                    order.reference_price,
                )
                price = float(order.reference_price)
            self._pos[order.symbol] = self._pos.get(order.symbol, 0.0) + order.qty
            self._cash -= order.qty * price
            self._filled.add(oid)
            fills.append(
                Fill(
                    client_order_id=oid,
                    symbol=order.symbol,
                    qty=order.qty,
                    price=price,
                    filled_bar=self._fill_bar,
                )
            )
        return fills


def load_local_bar_panels(
    symbols: Sequence[str],
    data_dir: str | Path = "data",
    *,
    interval: str = "1d",
    max_missing: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wide (close, volume, open) panels from the local parquet caches, offline.

    Concatenates every ``{symbol}_{interval}_*.parquet`` under ``data_dir``
    (the main range cache plus any delta caches), de-duplicated keep-last and
    sorted — the same union the incremental store would serve, without a
    network fetch. Replays need the *open* column for fills, which is why this
    does not reuse ``fetch_universe_panels`` (close/volume only).

    ``max_missing`` mirrors ``fetch_universe_panels``: the fraction of symbols
    allowed to have no local cache before failing loud (N7); tolerated misses
    are dropped with a warning naming them.
    """
    if not symbols:
        raise ValueError("empty symbol list — nothing to load")
    data_dir = Path(data_dir)
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    missing: list[str] = []
    for symbol in symbols:
        paths = sorted(data_dir.glob(f"{symbol}_{interval}_*.parquet"))
        if not paths:
            missing.append(symbol)
            continue
        bars = pd.concat([pd.read_parquet(p) for p in paths])
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        if bars.empty or "close" not in bars.columns or "open" not in bars.columns:
            missing.append(symbol)
            continue
        closes[symbol] = pd.to_numeric(bars["close"], errors="coerce")
        opens[symbol] = pd.to_numeric(bars["open"], errors="coerce")
        if "volume" in bars.columns:
            volumes[symbol] = pd.to_numeric(bars["volume"], errors="coerce")
    if missing:
        shown = missing[:20]
        tail = "…" if len(missing) > len(shown) else ""
        if len(missing) > max_missing * len(symbols):
            raise RuntimeError(
                f"no local bars for {len(missing)}/{len(symbols)} symbols "
                f"({len(missing) / len(symbols):.1%} > max_missing {max_missing:.1%}) "
                f"under {data_dir}: {shown}{tail}; fetch them first or raise max_missing (N7)"
            )
        logger.warning(
            "no local bars for %d/%d symbols (<= max_missing %.1f%%); replaying without them: %s%s",
            len(missing),
            len(symbols),
            100 * max_missing,
            shown,
            tail,
        )
    if not closes:
        raise RuntimeError(f"no local bars for any of the {len(symbols)} requested symbols (N7)")
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(columns=close.columns).sort_index()
    open_ = pd.DataFrame(opens).reindex(columns=close.columns).sort_index()
    return close, volume, open_


def align_replay_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    open_: pd.DataFrame,
    *,
    consensus: float = 0.95,
    clean_window: int = 400,
    keep: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict ragged per-name panels to a consensus calendar and clean names.

    A naive union of per-name vendor caches is ragged (vendor-holiday drift),
    and a strict full-history eligibility screen NaNs the whole book on the
    stray rows. The live venue serves ONE calendar for every name, so a replay
    mirrors that: keep rows where at least ``consensus`` of names have a close,
    then keep names with a complete close history over the trailing
    ``clean_window`` rows. ``keep`` names (e.g. an inherited cutover book that
    the mark step must price, N7) must survive the screen: one that does not
    raises loudly rather than silently vanishing from the replayed book.
    """
    rows = close.index[close.notna().sum(axis=1) >= consensus * close.shape[1]]
    close, volume, open_ = close.loc[rows], volume.loc[rows], open_.loc[rows]
    recent = close.tail(clean_window)
    clean = [s for s in close.columns if bool(recent[s].notna().all())]
    dropped_keep = [s for s in keep if s not in clean]
    if dropped_keep:
        raise ValueError(
            f"keep names {dropped_keep} lack a clean {clean_window}-bar close history — "
            f"the replay could not price them at the mark step (N7)"
        )
    return close[clean], volume[clean], open_[clean]


def replay_daily_cycles(
    signal_factory: Callable[[], Signal],
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    open_: pd.DataFrame,
    config: DailyBookConfig,
    run_dir: str | Path,
    *,
    start_bar: str | None = None,
    end_bar: str | None = None,
    initial_cash: float = 100_000.0,
    initial_positions: dict[str, float] | None = None,
    unrankable: Callable[[str], Collection[str]] | None = None,
    tranches: int = 1,
) -> tuple[list[DailyCycleResult], ReplayBroker]:
    """Run the real daily cycle over every bar in ``[start_bar, end_bar]``.

    ``unrankable`` threads the live driver's unrankable-mask seam
    (docs/bar_vendor_divergence.md §5) through every replayed cycle: a
    bar → symbols callable, consulted on refresh sessions only. A replay has
    no network, so callers inject a canned mapping (there is no offline event
    source to wire a CLI flag to); the default ``None`` is off — parity with
    every prior replay.

    ``tranches`` replays a tranched book (``run_daily_cycle(tranches=...)``,
    :mod:`prism.live.tranches`) on the epoch grid (base offset 0; the replay
    takes no seeded phase). The default 1 is the single book, as before.

    Each cycle *t*: arm bar *t*'s opens on the broker (the prior decision's
    N2 next-open fills), fit a fresh ``signal_factory()`` on the trailing
    panels through *t* (the live loop's own refit-per-run staleness policy —
    causal by construction, and free for a stateless node whose fit only
    validates), then ``run_daily_cycle`` on the truncated panels. The final
    bar's orders remain pending, exactly as a live loop ends its day.

    ``run_dir`` receives ``state.json`` / ``fills.jsonl`` / ``equity.jsonl``.
    It must not already hold loop state: a replay stream is a modeled-fill
    diagnostic and may never resume — or be mistaken for — a live ledger
    (see the module docstring), so an existing ``state.json`` refuses loudly
    rather than being adopted.

    Returns the per-cycle results and the broker (final positions/cash).
    """
    run_dir = Path(run_dir)
    if (run_dir / "state.json").exists():
        raise RuntimeError(
            f"{run_dir} already holds loop state — refusing to replay into an existing "
            f"run-dir (a replay stream must stay separate from any live ledger); "
            f"point --run-dir at a fresh directory or delete the old replay first"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    required = signal_factory().required_history
    dates = [str(d.date()) for d in close.index]
    bars = [
        (i, t)
        for i, t in enumerate(close.index)
        if (start_bar is None or dates[i] >= start_bar) and (end_bar is None or dates[i] <= end_bar)
    ]
    if not bars:
        raise ValueError(f"no bars in [{start_bar}, {end_bar}] — nothing to replay")
    first_pos = bars[0][0]
    if first_pos + 1 < required:
        raise ValueError(
            f"first replay bar {dates[first_pos]} has only {first_pos + 1} trailing rows "
            f"< required_history {required}; start the window later or extend the panel (N7)"
        )

    broker = ReplayBroker(initial_cash, initial_positions)
    ctx = LiveLoopContext(
        store=StateStore(run_dir / "state.json"),
        broker=broker,
        fills_ledger=run_dir / "fills.jsonl",
        equity_ledger=run_dir / "equity.jsonl",
        # The per-refresh decided book, same artifact the live loop persists —
        # the concordance-diagnostic object (replay book vs the research
        # backtest's target rows). Still a replay artifact: never calibration,
        # never the live concordance stream.
        targets_ledger=run_dir / "targets.jsonl",
    )
    logger.info(
        "replay: %d cycles %s -> %s over %d names (modeled next-open fills — diagnostic stream)",
        len(bars),
        dates[bars[0][0]],
        dates[bars[-1][0]],
        close.shape[1],
    )
    results: list[DailyCycleResult] = []
    for i, t in bars:
        sub_close = close.iloc[: i + 1]
        sub_volume = volume.iloc[: i + 1] if volume is not None else None
        broker.set_fill(open_.loc[t], dates[i])
        signal = signal_factory().fit(sub_close, sub_volume)
        results.append(
            run_daily_cycle(
                ctx,
                signal,
                sub_close,
                sub_volume,
                config,
                decision_bar=dates[i],
                unrankable=unrankable,
                tranches=tranches,
            )
        )
    return results, broker


# ---------------------------------------------------------------------------
# upgrade-diff: what an upgrade would decide differently (§3.11, stage 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionDecision:
    """One session's decision under one engine version.

    ``targets`` is the decided book — sid to weight, nonzero entries only,
    which is the form the live loop persists. ``orders`` is the signed share
    order the decision produced, or ``None`` for a source that records no order
    set at all; ``None`` and ``{}`` are different facts and the diff keeps them
    apart. ``refreshed`` says whether this session re-decided: a cadence hold
    is not a decision, and two engines that both hold have not disagreed.
    """

    bar: str
    refreshed: bool
    targets: Mapping[str, float] = field(default_factory=dict)
    orders: Mapping[str, float] | None = None
    equity: float | None = None


@dataclass(frozen=True)
class EngineDecisions:
    """What one engine version decided over a set of sessions.

    ``unavailable`` is the half of the record that makes the comparison
    honest: a session this engine could not be run over is named here with the
    reason, and never silently absent. A report over sessions nobody could
    compute must say so rather than reporting "no differences".
    """

    engine_version: str
    label: str
    decisions: Mapping[str, SessionDecision] = field(default_factory=dict)
    unavailable: Mapping[str, str] = field(default_factory=dict)
    source: str = ""
    notes: tuple[str, ...] = ()

    @property
    def bars(self) -> tuple[str, ...]:
        """Every session this side knows about, decided or not."""
        return tuple(sorted(set(self.decisions) | set(self.unavailable)))

    def describe(self) -> str:
        return f"{self.label} ({self.engine_version})" if self.label else self.engine_version

    @classmethod
    def unavailable_for(
        cls,
        engine_version: str,
        bars: Iterable[str],
        reason: str,
        *,
        label: str = "",
        source: str = "",
        notes: Sequence[str] = (),
    ) -> "EngineDecisions":
        """A side that could not be produced at all, with the reason on every bar.

        This is the honest object for "the candidate engine cannot be run over
        these sessions" — no local bar panel, no recorded capture — and it is
        what keeps :func:`upgrade_diff` from reporting an empty difference set
        over sessions it never compared.
        """
        return cls(
            engine_version=engine_version,
            label=label,
            decisions={},
            unavailable={str(bar): reason for bar in bars},
            source=source,
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class WeightChange:
    """One name's weight (or share) under each engine version."""

    sid: str
    baseline: float
    candidate: float

    @property
    def difference(self) -> float:
        return self.candidate - self.baseline

    def to_dict(self) -> dict[str, float | str]:
        return {
            "sid": self.sid,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "difference": self.difference,
        }


@dataclass(frozen=True)
class SessionDiff:
    """What the two engine versions would have done differently on one session."""

    bar: str
    comparable: bool
    reason: str | None = None
    baseline_refreshed: bool | None = None
    candidate_refreshed: bool | None = None
    entered: tuple[WeightChange, ...] = ()
    exited: tuple[WeightChange, ...] = ()
    moved: tuple[WeightChange, ...] = ()
    order_changes: tuple[WeightChange, ...] = ()
    orders_compared: bool = False

    @property
    def weight_changes(self) -> tuple[WeightChange, ...]:
        return self.entered + self.exited + self.moved

    @property
    def refresh_changed(self) -> bool:
        """Whether one version re-decided this session and the other held."""
        return (
            self.comparable
            and self.baseline_refreshed is not None
            and self.baseline_refreshed != self.candidate_refreshed
        )

    @property
    def differs(self) -> bool:
        return bool(self.weight_changes or self.order_changes or self.refresh_changed)

    @property
    def max_weight_difference(self) -> float:
        """The largest single-name weight move between the two versions."""
        return max((abs(change.difference) for change in self.weight_changes), default=0.0)

    @property
    def gross_weight_difference(self) -> float:
        """Summed absolute weight difference — the book distance between them."""
        return float(sum(abs(change.difference) for change in self.weight_changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar": self.bar,
            "comparable": self.comparable,
            "reason": self.reason,
            "baseline_refreshed": self.baseline_refreshed,
            "candidate_refreshed": self.candidate_refreshed,
            "refresh_changed": self.refresh_changed,
            "differs": self.differs,
            "entered": [c.to_dict() for c in self.entered],
            "exited": [c.to_dict() for c in self.exited],
            "moved": [c.to_dict() for c in self.moved],
            "orders_compared": self.orders_compared,
            "order_changes": [c.to_dict() for c in self.order_changes],
            "max_weight_difference": self.max_weight_difference,
            "gross_weight_difference": self.gross_weight_difference,
        }


@dataclass(frozen=True)
class UpgradeDiff:
    """What adopting one engine version in place of another would change.

    The object a review reads before an upgrade is adopted: per session, which
    names enter, which leave, which move and by how much, plus — separately and
    never folded in — the sessions that could not be compared and why. The two
    halves are kept apart on purpose. "No differences over the sessions we
    could run" and "no differences" are not the same claim, and only the first
    one is ever true of a report that could not run everything.
    """

    baseline: str
    candidate: str
    sessions: tuple[SessionDiff, ...] = ()
    notes: tuple[str, ...] = ()
    weight_tolerance: float = 0.0
    share_tolerance: float = 0.0

    @property
    def comparable(self) -> tuple[SessionDiff, ...]:
        return tuple(session for session in self.sessions if session.comparable)

    @property
    def incomparable(self) -> tuple[SessionDiff, ...]:
        return tuple(session for session in self.sessions if not session.comparable)

    @property
    def differing(self) -> tuple[SessionDiff, ...]:
        return tuple(session for session in self.comparable if session.differs)

    @property
    def complete(self) -> bool:
        """Whether every requested session was actually compared."""
        return bool(self.sessions) and not self.incomparable

    @property
    def identical(self) -> bool:
        """Whether every *compared* session decided the same thing.

        Read it with :attr:`complete`: identical over nothing is not agreement.
        """
        return not self.differing

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "tolerances": {"weight": self.weight_tolerance, "share": self.share_tolerance},
            "sessions_requested": len(self.sessions),
            "sessions_compared": len(self.comparable),
            "sessions_differing": len(self.differing),
            "complete": self.complete,
            "identical": self.identical,
            "notes": list(self.notes),
            "sessions": [session.to_dict() for session in self.sessions],
        }

    def summary(self) -> str:
        """A few lines an operator can read before deciding to adopt an upgrade."""
        lines = [
            f"upgrade-diff: {self.baseline} -> {self.candidate}",
            (
                f"{len(self.comparable)}/{len(self.sessions)} session(s) compared, "
                f"{len(self.differing)} with a different decision"
            ),
        ]
        for session in self.differing:
            lines.append(
                f"  {session.bar}: +{len(session.entered)} -{len(session.exited)} "
                f"~{len(session.moved)} names, max |Δw| {session.max_weight_difference:.6f}, "
                f"gross |Δw| {session.gross_weight_difference:.6f}"
                + (f", {len(session.order_changes)} order(s) differ" if session.orders_compared else "")
            )
        for session in self.incomparable:
            lines.append(f"  {session.bar}: NOT COMPARED — {session.reason}")
        if self.incomparable:
            lines.append(
                "This report does not say the upgrade changes nothing: it says nothing at all about the "
                "sessions above."
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _weight_diff(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    tolerance: float,
) -> tuple[tuple[WeightChange, ...], tuple[WeightChange, ...], tuple[WeightChange, ...]]:
    """(entered, exited, moved) for two books, at one tolerance."""
    entered: list[WeightChange] = []
    exited: list[WeightChange] = []
    moved: list[WeightChange] = []
    for sid in sorted(set(baseline) | set(candidate)):
        before = float(baseline.get(sid, 0.0))
        after = float(candidate.get(sid, 0.0))
        if abs(after - before) <= tolerance:
            continue
        change = WeightChange(sid=sid, baseline=before, candidate=after)
        if abs(before) <= tolerance:
            entered.append(change)
        elif abs(after) <= tolerance:
            exited.append(change)
        else:
            moved.append(change)
    return tuple(entered), tuple(exited), tuple(moved)


def upgrade_diff(
    baseline: EngineDecisions,
    candidate: EngineDecisions,
    *,
    bars: Sequence[str] | None = None,
    weight_tolerance: float = 1e-9,
    share_tolerance: float = 1e-9,
) -> UpgradeDiff:
    """Compare two engine versions' decisions over a set of recorded sessions.

    This is the mechanism for reviewing an upgrade without maintaining a fork:
    the sessions are the record, the two sides are what each engine version
    decided over them, and the report says — per session — which names enter,
    which leave, which move and by how much, and which sessions could not be
    compared at all.

    It compares *decisions*, never outcomes. Fill prices, realised equity and
    execution quality belong to the venue and the day, not to the engine
    version, so a difference in them is not evidence about an upgrade. Order
    quantities are compared only when both sides record an order set; a side
    that records none leaves ``orders_compared`` false rather than reporting
    that every order changed.

    ``bars`` defaults to the union of both sides' sessions, so a session that
    only one side has is reported as incomparable rather than dropped.
    """
    requested = list(bars) if bars is not None else sorted(set(baseline.bars) | set(candidate.bars))
    diffs: list[SessionDiff] = []
    for bar in requested:
        bar = str(bar)
        left = baseline.decisions.get(bar)
        right = candidate.decisions.get(bar)
        if left is None or right is None:
            missing = []
            if left is None:
                missing.append(
                    f"{baseline.describe()}: "
                    + str(baseline.unavailable.get(bar, "no decision recorded for this session"))
                )
            if right is None:
                missing.append(
                    f"{candidate.describe()}: "
                    + str(candidate.unavailable.get(bar, "no decision recorded for this session"))
                )
            diffs.append(SessionDiff(bar=bar, comparable=False, reason="; ".join(missing)))
            continue
        entered, exited, moved = _weight_diff(left.targets, right.targets, weight_tolerance)
        orders_compared = left.orders is not None and right.orders is not None
        order_changes: tuple[WeightChange, ...] = ()
        if orders_compared:
            a, b, c = _weight_diff(left.orders or {}, right.orders or {}, share_tolerance)
            order_changes = a + b + c
        diffs.append(
            SessionDiff(
                bar=bar,
                comparable=True,
                baseline_refreshed=left.refreshed,
                candidate_refreshed=right.refreshed,
                entered=entered,
                exited=exited,
                moved=moved,
                order_changes=order_changes,
                orders_compared=orders_compared,
            )
        )
    return UpgradeDiff(
        baseline=baseline.describe(),
        candidate=candidate.describe(),
        sessions=tuple(diffs),
        notes=tuple(dict.fromkeys(baseline.notes + candidate.notes)),
        weight_tolerance=weight_tolerance,
        share_tolerance=share_tolerance,
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _recorded_orders(fills: Sequence[Mapping[str, Any]], unfilled: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The parent order set one recorded session submitted, sid -> signed shares.

    The run directory records what each order *was* in two places: the unfilled
    ledger carries ``order_qty`` for every order that left a residual, and the
    fills ledger carries the executed quantity, which for an order that filled
    whole is the same number. Every order therefore has a row in one of them.

    Completion-sweep orders are excluded, because a sweep order is a residual of
    a decision rather than a decision. They are identified structurally — a
    sweep child's id is its parent's id plus one suffix segment
    (``loop.sweep_pending``) — rather than by matching a literal ``:S1``, since
    the suffix is the caller's and the client id prefix is the book's, and
    counting colons gets a prefix-less run exactly backwards.
    """
    rows = [*unfilled, *fills]
    recorded_ids = {str(row.get("client_order_id", "")) for row in rows}

    def is_sweep_child(client_order_id: str) -> bool:
        parent, _, _suffix = client_order_id.rpartition(":")
        return bool(parent) and parent in recorded_ids

    orders: dict[str, float] = {}
    for row in unfilled:
        if is_sweep_child(str(row.get("client_order_id", ""))):
            continue
        orders[str(row["symbol"])] = float(row["order_qty"])
    for row in fills:
        if is_sweep_child(str(row.get("client_order_id", ""))):
            continue
        orders.setdefault(str(row["symbol"]), float(row["qty"]))
    return orders


def recorded_decisions(
    capture_dir: Path | str,
    *,
    engine_version: str | None = None,
    label: str = "recorded",
    bars: Sequence[str] | None = None,
) -> EngineDecisions:
    """The decisions a recorded run made, read from a session capture directory.

    The capture is a run directory's own ledgers plus the ``manifest.json`` that
    says which sessions were captured, which of them decided, which engine build
    traded them, and what the capture is *not* able to support — the manifest's
    caveats travel onto the report as notes, because a limitation recorded in a
    fixture and dropped by the tool that reads it is a limitation nobody sees.

    A session the manifest lists as a hold is a decision-bearing fact in its own
    right: the engine held. It is recorded with ``refreshed=False`` and an empty
    book, not as a missing session.
    """
    capture_dir = Path(capture_dir)
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{capture_dir} holds no manifest.json; a session capture without its manifest cannot say which "
            "sessions it covers or which engine build traded them (N7)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if engine_version is None:
        traded_commit = str((manifest.get("source") or {}).get("traded_commit") or "")
        engine_version = f"git:{traded_commit}" if traded_commit else "unknown"

    targets_rows = {str(row["decision_bar"]): row for row in _read_jsonl(capture_dir / "targets.jsonl")}
    fills_rows = _read_jsonl(capture_dir / "fills.jsonl")
    unfilled_rows = _read_jsonl(capture_dir / "unfilled.jsonl")
    has_order_record = bool(fills_rows or unfilled_rows)

    sessions = manifest.get("sessions") or []
    listed = [str(session["bar"]) for session in sessions]
    wanted = [str(bar) for bar in bars] if bars is not None else listed
    kinds = {str(session["bar"]): str(session.get("kind", "")) for session in sessions}

    decisions: dict[str, SessionDecision] = {}
    unavailable: dict[str, str] = {}
    for bar in wanted:
        if bar not in kinds:
            unavailable[bar] = f"the capture at {capture_dir} does not cover this session"
            continue
        row = targets_rows.get(bar)
        if row is None:
            if kinds[bar] == "hold":
                decisions[bar] = SessionDecision(bar=bar, refreshed=False, targets={}, orders={})
            else:
                unavailable[bar] = (
                    "the manifest calls this a decision-bearing session and the targets ledger holds no row " "for it"
                )
            continue
        session_fills = [f for f in fills_rows if str(f.get("decision_bar")) == bar]
        session_unfilled = [u for u in unfilled_rows if str(u.get("decision_bar")) == bar]
        decisions[bar] = SessionDecision(
            bar=bar,
            refreshed=True,
            targets={str(sid): float(weight) for sid, weight in (row.get("targets") or {}).items()},
            orders=_recorded_orders(session_fills, session_unfilled) if has_order_record else None,
            equity=None if row.get("equity") is None else float(row["equity"]),
        )
    return EngineDecisions(
        engine_version=str(engine_version),
        label=label,
        decisions=decisions,
        unavailable=unavailable,
        source=str(capture_dir),
        notes=tuple(str(note) for note in (manifest.get("caveats") or ())),
    )


def replayed_decisions(
    signal_factory: Callable[[], Signal],
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    open_: pd.DataFrame,
    config: DailyBookConfig,
    run_dir: str | Path,
    *,
    bars: Sequence[str],
    engine_version: str | None = None,
    label: str = "replay",
    initial_cash: float = 100_000.0,
    initial_positions: dict[str, float] | None = None,
    unrankable: Callable[[str], Collection[str]] | None = None,
) -> EngineDecisions:
    """Run the daily cycle over ``bars`` and record what it decided.

    The candidate side of an :func:`upgrade_diff` when the engine under review
    is the one in this process. A session the panel does not reach is recorded
    as unavailable with the reason rather than skipped, because "the new engine
    decided nothing different here" and "the new engine was never run here" are
    the two answers an upgrade review must not confuse.

    Modeled fills, as everywhere in this module: the decided book is the
    comparable object, and the replayed equity and fill prices are not.
    """
    wanted = [str(bar) for bar in bars]
    if not wanted:
        raise ValueError("no sessions requested — nothing to replay")
    covered = {str(stamp.date()) for stamp in close.index}
    missing = [bar for bar in wanted if bar not in covered]
    if engine_version is None:
        from prism.spec import engine_version as installed_engine_version

        engine_version = installed_engine_version()
    if len(missing) == len(wanted):
        return EngineDecisions.unavailable_for(
            str(engine_version),
            wanted,
            (
                f"no bar panel covers this session: the panel supplied spans "
                f"{min(covered, default='nothing')}..{max(covered, default='nothing')}"
            ),
            label=label,
            source=str(run_dir),
        )
    playable = [bar for bar in wanted if bar in covered]
    results, _broker = replay_daily_cycles(
        signal_factory,
        close,
        volume,
        open_,
        config,
        run_dir,
        start_bar=min(playable),
        end_bar=max(playable),
        initial_cash=initial_cash,
        initial_positions=initial_positions,
        unrankable=unrankable,
    )
    refresh_bars = {str(row["refresh_bar"]) for row in read_targets_ledger(Path(run_dir) / "targets.jsonl")}
    decisions: dict[str, SessionDecision] = {}
    for result in results:
        bar = str(result.decision_bar)
        if bar not in wanted:
            continue  # a bar inside the replayed span that the caller did not ask about
        weights = result.target_weights
        decisions[bar] = SessionDecision(
            bar=bar,
            refreshed=bar in refresh_bars,
            targets={
                str(sid): float(weight)
                for sid, weight in weights.items()
                if np.isfinite(float(weight)) and float(weight) != 0.0
            },
            orders={str(order.symbol): float(order.qty) for order in result.submitted_orders},
            equity=float(result.equity),
        )
    unavailable = {bar: "no bar panel covers this session" for bar in wanted if bar not in covered}
    unavailable.update(
        {bar: "the replay produced no cycle for this session" for bar in playable if bar not in decisions}
    )
    return EngineDecisions(
        engine_version=str(engine_version),
        label=label,
        decisions=decisions,
        unavailable=unavailable,
        source=str(run_dir),
    )


def upgrade_diff_report(
    sessions: Path | str,
    *,
    baseline: Path | str | None = None,
    candidate: Path | str | None = None,
    bars: Sequence[str] | None = None,
    weight_tolerance: float = 1e-9,
    share_tolerance: float = 1e-9,
) -> UpgradeDiff:
    """The offline ``upgrade-diff``: two captures, one set of recorded sessions.

    ``sessions`` is the capture whose manifest defines which sessions are under
    review; ``baseline`` and ``candidate`` are the captures holding what each
    engine version decided over them, defaulting to ``sessions`` for the
    baseline and to *nothing* for the candidate. A missing candidate is not an
    error — it is the ordinary state before the upgrade has been run over these
    sessions — and it produces a report in which every session is named as not
    compared, with the reason.
    """
    sessions_dir = Path(sessions)
    left = recorded_decisions(baseline if baseline is not None else sessions_dir, label="baseline")
    requested = list(bars) if bars is not None else list(left.bars)
    if candidate is None:
        right = EngineDecisions.unavailable_for(
            "unknown",
            requested,
            (
                "no candidate capture was supplied: the upgraded engine has not been run over these "
                "recorded sessions, so there is nothing to compare against"
            ),
            label="candidate",
        )
    else:
        right = recorded_decisions(candidate, label="candidate", bars=requested)
    return upgrade_diff(
        left,
        right,
        bars=requested,
        weight_tolerance=weight_tolerance,
        share_tolerance=share_tolerance,
    )
