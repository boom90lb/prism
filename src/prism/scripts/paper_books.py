"""Book assembly registry for the paper-loop CLI (SPEC §13 R2; B1/T0 books).

``prism.scripts.paper_loop``'s own rule is that the shell holds no logic
worth testing — yet by 2026-07-29 its ``main`` had grown a 300-line braid of
per-book if/elif (signal factory, construction config, order-id namespace,
cadence and missing-bar defaults) that nothing exercised offline. This
module is where that logic now lives: one declarative :class:`BookSpec` per
tradable book, a flat :data:`BOOKS` registry the shell dispatches through,
and pure helpers for the remaining assembly decisions (universe resolution,
profile pinning, safety-rail derivation). Everything here is testable
without credentials (tests/test_paper_books.py).

Ratified-config discipline: the specs *reproduce* the CLI's existing
defaults — the momentum spec under CLI defaults must stay field-identical
to ``CERTIFIED_B1_PAPER_CONFIG`` (G6), which the test suite asserts via
``assert_research_paper_bit_identity``. Changing a default here is a
discovery event, not a refactor.

Two of the three books also carry a *specification file*
(``BookSpec.specification``, loadable through
:func:`load_book_specification`): ``examples/momentum/reference.toml`` and
``examples/trend/reference.toml`` declare the same universe, cadence,
component parameters, rank-to-weight rule and eligibility screen this
registry builds, in the engine's own configuration vocabulary
(``prism.spec``). They are the declaration of record, asserted
field-by-field against the registry in
``tests/test_paper_books_specifications.py`` — not a second construction
path. The CLI keeps building the book from the callables below, because a
fixed configuration table cannot carry what the flags vary per run
(``horizon_bars`` tracks ``--decision-every``, the caps track
``--max-gross`` / ``--max-symbol-weight``) nor what the cycle computes at
run time (``MomentumSignalNode``'s ``membership_mask``). The ensemble book
has no specification at all: its ``directional`` construction — a linear
score-to-weight map with a ``position_size`` slope — is not one of the four
rank-to-weight rules, and a ``standardized_forecast`` component is not a
kind position-space combination accepts.

Its node is also no longer in the wheel. At stage 6 the learned ensemble moved
to the optional ``prism-ensemble`` distribution (engine specification section 2,
`signal/ensemble_node.py`, Removal), so the registry still carries the book and
``_ensemble_signal`` imports the package when the book is dispatched. Every
other book is unaffected by whether it is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import pandas as pd

from prism.live.daily import DailyBookConfig
from prism.live.safety import SafetyConfig
from prism.residual.factors import ResidualStatArbConfig
from prism.signal.base import Signal
from prism.signal.momentum_node import MomentumSignalNode
from prism.signal.trend_node import TREND_V1_UNIVERSE, TrendSignalNode

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from prism.spec import Specification
    from prism.workspace import Workspace


@dataclass(frozen=True)
class BookParams:
    """Every CLI-varied construction knob, book-agnostic; specs read what
    they need. Defaults mirror the CLI defaults exactly — the momentum spec
    under these defaults is the certified B1 paper config (G6)."""

    lookback_bars: int = 252
    skip_bars: int = 21
    decile: float = 0.10
    vol_ewma_bars: int = 63
    horizon_bars: int = 5
    decision_every: int | None = None  # None -> the spec's cadence default
    position_size: float = 0.05
    max_gross: float = 1.0
    max_symbol_abs_weight: float = 0.10
    no_trade_band: float = 0.0
    max_participation: float | None = None
    min_order_notional: float = 1.0
    whole_shares: bool = True


@dataclass(frozen=True)
class BookSpec:
    """One tradable book: how the CLI assembles its signal and construction.

    ``order_id_prefix`` namespaces client order ids per book (two books
    sharing one venue account with the bare ``{bar}:{symbol}`` scheme
    silently substitute each other's same-bar orders).
    ``default_decision_every``/``default_max_missing``/``panel_start_years``
    are the CLI defaults the operator may override per run.

    ``specification`` is the workspace-relative path of the book's declaration
    of record, or ``None`` for a book the configuration vocabulary cannot
    express. It names a file, not a construction path: the callables above
    remain the only thing that builds a signal or a ``DailyBookConfig``.
    """

    name: str
    order_id_prefix: str
    default_decision_every: int
    default_max_missing: float
    panel_start_years: int | None
    build_signal: Callable[[BookParams, int, pd.DataFrame | None], Signal]
    build_config: Callable[[BookParams, int], DailyBookConfig]
    describe: Callable[[BookParams, int, int, Signal], str]
    default_universe: tuple[str, ...] = field(default=())
    specification: str | None = None

    def decision_every(self, params: BookParams) -> int:
        return (
            params.decision_every
            if params.decision_every is not None
            else self.default_decision_every
        )


# --------------------------------------------------------------------- books


def _momentum_signal(params: BookParams, decision_every: int, score_mask: pd.DataFrame | None) -> Signal:
    return MomentumSignalNode(
        ResidualStatArbConfig(),
        lookback_bars=params.lookback_bars,
        skip_bars=params.skip_bars,
        horizon_bars=decision_every,
        membership_mask=score_mask,
    )


def _momentum_config(params: BookParams, decision_every: int) -> DailyBookConfig:
    return DailyBookConfig(
        book="decile_neutral",
        decile=params.decile,
        decision_every=decision_every,
        max_gross=params.max_gross,
        max_symbol_abs_weight=params.max_symbol_abs_weight,
        no_trade_band=params.no_trade_band,
        max_participation=params.max_participation,
        min_order_notional=params.min_order_notional,
        whole_shares=params.whole_shares,
    )


def _momentum_describe(params: BookParams, decision_every: int, n_symbols: int, signal: Signal) -> str:
    return (
        f"book=momentum: 12-1 decile L/S, lookback={params.lookback_bars} "
        f"skip={params.skip_bars} decile={params.decile:.2f} cadence={decision_every}, "
        f"{n_symbols} symbols"
    )


def _trend_signal(params: BookParams, decision_every: int, score_mask: pd.DataFrame | None) -> Signal:
    return TrendSignalNode(
        lookback_bars=params.lookback_bars,
        skip_bars=params.skip_bars,
        horizon_bars=decision_every,
    )


def _trend_config(params: BookParams, decision_every: int) -> DailyBookConfig:
    # Default max_symbol_weight 0.10 is tight for a 10-name inv-vol book; the
    # operator may raise it via flag. Participation defaults ON (0.05) for the
    # ETF book unless the CLI set a value explicitly.
    return DailyBookConfig(
        book="inverse_vol",
        vol_ewma_bars=params.vol_ewma_bars,
        decision_every=decision_every,
        max_gross=params.max_gross,
        max_symbol_abs_weight=params.max_symbol_abs_weight,
        no_trade_band=params.no_trade_band,
        max_participation=params.max_participation if params.max_participation is not None else 0.05,
        min_order_notional=params.min_order_notional,
        whole_shares=params.whole_shares,
    )


def _trend_describe(params: BookParams, decision_every: int, n_symbols: int, signal: Signal) -> str:
    return (
        f"book=trend: 12-1 TSMOM inv-vol, lookback={params.lookback_bars} "
        f"skip={params.skip_bars} vol_ewma={params.vol_ewma_bars} cadence={decision_every}, "
        f"{n_symbols} symbols"
    )


def _ensemble_signal(params: BookParams, decision_every: int, score_mask: pd.DataFrame | None) -> Signal:
    """Construct the learned ensemble node out of the optional strategy package.

    The import is here and not at module scope on purpose. The node left the
    wheel at stage 6 (engine specification section 2, `signal/ensemble_node.py`,
    Removal) and lives in the separate ``prism-ensemble`` distribution, so an
    engine install without it must still dispatch `momentum` and `trend` --
    which a module-scope import would break for every book at once. An operator
    who asks for this book without the package gets told what to install.
    """
    try:
        from prism_ensemble import EnsembleNodeConfig, EnsembleSignalNode
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the package
        raise ModuleNotFoundError(
            "the ensemble book's signal node is the optional prism-ensemble distribution, "
            "which is not installed (`pip install prism-ensemble`); the momentum and trend "
            "books do not need it"
        ) from exc
    return EnsembleSignalNode(EnsembleNodeConfig(horizon_bars=params.horizon_bars))


def _ensemble_config(params: BookParams, decision_every: int) -> DailyBookConfig:
    # decision_every is threaded through (pre-registry it was silently dropped
    # for this book: --decision-every N --book ensemble constructed at the
    # default daily cadence). The default is still 1, so default behavior is
    # unchanged.
    return DailyBookConfig(
        position_size=params.position_size,
        decision_every=decision_every,
        max_gross=params.max_gross,
        max_symbol_abs_weight=params.max_symbol_abs_weight,
        no_trade_band=params.no_trade_band,
        max_participation=params.max_participation,
        min_order_notional=params.min_order_notional,
        whole_shares=params.whole_shares,
    )


def _ensemble_describe(params: BookParams, decision_every: int, n_symbols: int, signal: Signal) -> str:
    return (
        f"book=ensemble: {len(signal.fitted_symbols_)} symbols, "  # type: ignore[attr-defined]
        f"weights {signal.weight_basis_}"  # type: ignore[attr-defined]
    )


BOOKS: dict[str, BookSpec] = {
    "momentum": BookSpec(
        name="momentum",
        order_id_prefix="mom:",
        default_decision_every=21,
        default_max_missing=0.10,
        panel_start_years=3,
        build_signal=_momentum_signal,
        build_config=_momentum_config,
        describe=_momentum_describe,
        specification="examples/momentum/reference.toml",
    ),
    "trend": BookSpec(
        name="trend",
        order_id_prefix="trd:",
        default_decision_every=21,
        default_max_missing=0.10,
        panel_start_years=3,
        build_signal=_trend_signal,
        build_config=_trend_config,
        describe=_trend_describe,
        default_universe=tuple(TREND_V1_UNIVERSE),
        specification="examples/trend/reference.toml",
    ),
    "ensemble": BookSpec(
        name="ensemble",
        order_id_prefix="",
        default_decision_every=1,
        default_max_missing=0.0,
        panel_start_years=None,
        build_signal=_ensemble_signal,
        build_config=_ensemble_config,
        describe=_ensemble_describe,
    ),
}


# ------------------------------------------------------- specification files


def load_book_specification(book: str, *, workspace: "Workspace | None" = None) -> "Specification":
    """Load a book's declaration of record through the specification loader.

    The registry entry names a workspace-relative file; this resolves it
    against ``workspace`` (default: the resolved workspace) and hands it to
    ``prism.spec.load_spec``, which parses it, resolves its defaults, imports
    and instantiates its component, and can compute its ``config_id``.

    Raising is the point for a book with no ``specification``: the ensemble
    book's ``directional`` construction has no rank-to-weight rule and its
    component emits a ``standardized_forecast``, which position-space
    combination does not accept, so there is no file to load rather than a
    file that misdeclares it.

    ``prism.spec`` is imported here rather than at module import: the live
    paper path pays for the specification loader only when it asks for one.
    """
    try:
        spec = BOOKS[book]
    except KeyError:
        raise KeyError(f"unknown book {book!r}; registered books are {sorted(BOOKS)}") from None
    if spec.specification is None:
        raise ValueError(
            f"book {book!r} has no specification file: its construction is not expressible in the "
            "configuration vocabulary (prism.spec.TO_WEIGHTS_RULES has no rule for the directional "
            "book's position_size slope). Build it from the registry entry instead."
        )
    from prism.spec import load_spec
    from prism.workspace import resolve_workspace

    ws = workspace if workspace is not None else resolve_workspace()
    return load_spec(ws.root / spec.specification, workspace=ws)


# ----------------------------------------------------------------- assembly


def apply_certified_paper_pin(params: BookParams, pin: DailyBookConfig) -> BookParams:
    """The research_paper profile's construction pin (G6, tighten-nothing).

    Returns ``params`` with every certified construction field replaced by
    the pin's value; ``whole_shares`` is governed separately by the enforced
    ``--tif opg``. The resulting momentum config must satisfy
    ``assert_research_paper_bit_identity`` — the caller asserts it.
    """
    return replace(
        params,
        decile=pin.decile,
        decision_every=pin.decision_every,
        max_gross=pin.max_gross,
        max_symbol_abs_weight=pin.max_symbol_abs_weight,
        no_trade_band=pin.no_trade_band,
        max_participation=pin.max_participation,
        min_order_notional=pin.min_order_notional,
    )


def resolve_cli_universe(
    universe_file: Path | None,
    symbols_arg: str | None,
    spec: BookSpec,
) -> list[str]:
    """The configured universe, by CLI precedence: file > flag > spec default."""
    if universe_file is not None:
        from prism.io.universe_file import load_universe_symbols

        return load_universe_symbols(universe_file)
    if symbols_arg:
        return [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    if spec.default_universe:
        # Pinned universe (e.g. trend_v1, docs/trend_design.md §1); no free choice.
        return list(spec.default_universe)
    raise SystemExit("provide --symbols or --universe-file")


def default_panel_start(spec: BookSpec) -> str | None:
    """The spec's default panel start (None lets the loader/store decide).

    ~3y is ample for the 252-bar lookback (+ vol warmup for trend) and keeps
    the universe-scale batch fetch to a handful of pages.
    """
    if spec.panel_start_years is None:
        return None
    return (pd.Timestamp.now() - pd.DateOffset(years=spec.panel_start_years)).strftime("%Y-%m-%d")


def valuation_masked_membership(
    close: pd.DataFrame, score_universe: Sequence[str], has_extras: bool
) -> pd.DataFrame | None:
    """Momentum membership mask when valuation-only extras ride the panel.

    Extras are masked ineligible: NaN-scored names get an explicit 0.0 from
    the decile construct (its explicit-flat pin), so a departed-but-held name
    exits at the next refresh and cannot re-enter. ``None`` (no extras) keeps
    the unmasked path bit-identical.
    """
    if not has_extras:
        return None
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    mask.loc[:, [s for s in score_universe if s in close.columns]] = True
    return mask


def build_safety_config(
    *,
    run_dir: Path,
    kill_switch_arg: Path | None,
    max_drawdown: float,
    max_order_fraction: float | None,
    max_symbol_abs_weight: float,
    max_orders: int | None,
    n_symbols: int,
) -> SafetyConfig:
    """Safety rails from CLI flags: inert at the book's normal state, loud at
    pathology. The notional bound derives from the construction cap (2x is
    slack for whole-share rounding), the order-count bound from the
    universe's geometric ceiling; both catch order-of-magnitude corruption,
    not strategy behavior. 'off'/0 disables a rail explicitly.
    """
    kill_switch: Path | None
    if kill_switch_arg is None:
        kill_switch = run_dir / "KILL_SWITCH"
    elif str(kill_switch_arg) == "off":
        kill_switch = None
    else:
        kill_switch = kill_switch_arg
    resolved_fraction = max_order_fraction if max_order_fraction is not None else 2.0 * max_symbol_abs_weight
    resolved_orders = max_orders if max_orders is not None else 2 * n_symbols + 10
    return SafetyConfig(
        kill_switch=kill_switch,
        max_drawdown=max_drawdown if max_drawdown > 0 else None,
        max_order_fraction=resolved_fraction if resolved_fraction > 0 else None,
        max_orders=resolved_orders if resolved_orders > 0 else None,
    )
