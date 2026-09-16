"""The paper-loop books against their specification files (offline).

``examples/momentum/reference.toml`` and ``examples/trend/reference.toml`` are
the declaration of record for the two books the CLI can express in the engine's
configuration vocabulary. Nothing on the live path reads them, which is exactly
why they need a test: a declaration that drifts from the registry it describes
is worse than no declaration, because it reads as authoritative.

So every test here builds the book *both ways* — through
``prism.scripts.paper_books.BOOKS`` under CLI defaults, and through
``prism.spec.load_spec`` over the example file — and compares field by field,
ending with a bitwise comparison of the two components' scores on one panel.
The registry stays the construction path; these assertions are what keep the
files honest.

The five ``[policy]`` files are written here as fixtures. The loader requires
them to exist and hashes their content into ``config_id``; their schemas belong
to sections 3.6, 3.7, 3.9 and 3.11 and to the project template, not to this
comparison.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from prism.residual.factors import ResidualStatArbConfig
from prism.scripts.paper_books import BOOKS, BookParams, load_book_specification
from prism.scripts.paper_loop import _parse_args
from prism.signal.trend_node import TREND_V1_UNIVERSE
from prism.spec import ELIGIBILITY_DEFAULTS, Eligibility
from prism.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The `[policy]` files every example specification names, relative to the
#: workspace root. Content is irrelevant to this file's assertions.
POLICY_FILES = (
    "profiles/paper.toml",
    "costs/retail_v1.toml",
    "profiles/execution_alpaca.toml",
    "policies/monitoring.toml",
    "policies/recovery.toml",
)

#: Books whose construction the configuration vocabulary cannot express, and
#: which therefore carry no specification file. Widening this set is a decision,
#: not a refactor: it means a tradable book has no declaration of record.
BOOKS_WITHOUT_SPECIFICATION = {"ensemble"}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """A workspace holding the repository's ``examples/`` and policy fixtures."""
    root = tmp_path_factory.mktemp("book_specs")
    shutil.copytree(REPO_ROOT / "examples", root / "examples")
    for relative in POLICY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture stand-in for {relative}\n", encoding="utf-8")
    return Workspace(root)


def _panel(symbols: list[str], bars: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A closed-form panel long enough for a 252-bar lookback and the screen.

    No RNG: every close is a deterministic function of (bar, column), priced
    above ``min_price`` and turning over well above the dollar-volume floor, so
    the eligibility screen passes for every name from bar 253 onward.
    """
    index = pd.bdate_range("2023-01-02", periods=bars, tz="America/New_York")
    close = pd.DataFrame(
        {sym: [50.0 + 3.0 * col + t * (0.05 + 0.01 * col) for t in range(bars)] for col, sym in enumerate(symbols)},
        index=index,
        dtype=float,
    )
    volume = pd.DataFrame(1_000_000.0, index=index, columns=symbols, dtype=float)
    return close, volume


def _scores(signal, close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    return signal.fit(close, volume).score(close, volume)


# ---------------------------------------------------------------------------
# Which books carry a declaration at all
# ---------------------------------------------------------------------------


def test_only_the_inexpressible_book_lacks_a_specification() -> None:
    missing = {name for name, spec in BOOKS.items() if spec.specification is None}
    assert missing == BOOKS_WITHOUT_SPECIFICATION


def test_every_declared_specification_file_exists_in_the_repository() -> None:
    for name, spec in BOOKS.items():
        if spec.specification is None:
            continue
        assert (REPO_ROOT / spec.specification).is_file(), f"{name} names a missing specification"


def test_ensemble_is_refused_rather_than_misdeclared() -> None:
    # The directional book maps a standardized forecast linearly to a weight
    # with a position_size slope. That is not one of the four rank-to-weight
    # rules, and position-space combination takes rank_score or target_weights
    # only, so no file can declare this book truthfully.
    with pytest.raises(ValueError, match="not expressible in the configuration vocabulary"):
        load_book_specification("ensemble")


def test_unknown_book_names_the_registry() -> None:
    with pytest.raises(KeyError, match="unknown book"):
        load_book_specification("no_such_book")


# ---------------------------------------------------------------------------
# Momentum: registry vs specification, field by field
# ---------------------------------------------------------------------------


def test_momentum_specification_declares_the_registry_defaults(workspace: Workspace) -> None:
    loaded = load_book_specification("momentum", workspace=workspace)
    book = BOOKS["momentum"]
    params = BookParams()
    cadence = book.decision_every(params)
    config = book.build_config(params, cadence)
    component = loaded.strategy.components[0]

    assert loaded.strategy.combination == "position_space"
    assert loaded.construction.kind == "heuristic"
    assert loaded.strategy.renormalize is False
    assert len(loaded.strategy.components) == 1

    # Cadence: one number in the registry, three fields in the file.
    assert cadence == book.default_decision_every == 21
    assert component.horizon_bars == cadence
    assert component.update_every_bars == cadence
    assert component.valid_for_bars == cadence
    assert config.decision_every == cadence

    # Constructor arguments, exactly what `_momentum_signal` passes.
    assert dict(component.params) == {
        "lookback_bars": params.lookback_bars,
        "skip_bars": params.skip_bars,
        "horizon_bars": cadence,
    }

    # Rank-to-weight rule and its parameters against the constructed book.
    assert component.to_weights == "decile_long_short"
    assert dict(component.to_weights_params) == {
        "decile": config.decile,
        "max_gross": config.max_gross,
        "max_symbol_abs_weight": config.max_symbol_abs_weight,
    }
    assert config.book == "decile_neutral"


def test_momentum_specification_eligibility_is_the_screen_the_node_applies(workspace: Workspace) -> None:
    # The declared screen must be the one `compute_eligibility` enforces under
    # the default `ResidualStatArbConfig` the registry hands the node — a
    # declaration that overstates the screen is the failure this catches.
    loaded = load_book_specification("momentum", workspace=workspace)
    screen = loaded.strategy.eligibility
    config = ResidualStatArbConfig()

    assert screen.min_price == config.min_price
    assert screen.min_median_dollar_volume == config.min_median_dollar_volume
    assert screen.dollar_volume_window == config.dollar_volume_window
    assert screen.return_history_bars == config.corr_window
    # No asset class is excluded because no ETF symbols are configured; the two
    # facts have to move together.
    assert screen.exclude_asset_classes == ()
    assert config.etf_symbols == ()


def test_momentum_specification_component_scores_identically(workspace: Workspace) -> None:
    symbols = [f"SYM{i:02d}" for i in range(30)]
    close, volume = _panel(symbols)
    book = BOOKS["momentum"]
    params = BookParams()

    from_registry = book.build_signal(params, book.decision_every(params), None)
    from_specification = load_book_specification("momentum", workspace=workspace).component("momentum").component

    assert from_specification.horizon_bars == from_registry.horizon_bars
    assert from_specification.required_history == from_registry.required_history
    pd.testing.assert_series_equal(
        _scores(from_specification, close, volume),
        _scores(from_registry, close, volume),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Trend: registry vs specification, field by field
# ---------------------------------------------------------------------------


def test_trend_specification_declares_the_registry_defaults(workspace: Workspace) -> None:
    loaded = load_book_specification("trend", workspace=workspace)
    book = BOOKS["trend"]
    params = BookParams()
    cadence = book.decision_every(params)
    config = book.build_config(params, cadence)
    component = loaded.strategy.components[0]

    assert loaded.strategy.combination == "position_space"
    assert loaded.construction.kind == "heuristic"
    assert len(loaded.strategy.components) == 1

    # The pinned universe is declared inline, and it is the registry's default.
    assert loaded.strategy.universe_list == book.default_universe == tuple(TREND_V1_UNIVERSE)
    assert loaded.strategy.universe_index is None
    assert loaded.strategy.universe_as_of is not None
    assert loaded.strategy.universe_registration is not None

    assert cadence == book.default_decision_every == 21
    assert component.horizon_bars == cadence
    assert component.update_every_bars == cadence
    assert component.valid_for_bars == cadence
    assert config.decision_every == cadence

    assert dict(component.params) == {
        "lookback_bars": params.lookback_bars,
        "skip_bars": params.skip_bars,
        "horizon_bars": cadence,
    }

    assert component.to_weights == "sign_inverse_vol"
    assert dict(component.to_weights_params) == {
        "vol_ewma_bars": config.vol_ewma_bars,
        # `_construct_raw_book` passes no override, so the construct's own
        # default is the value the live book annualizes by.
        "annualization_bars": 252,
        "max_gross": config.max_gross,
        "max_symbol_abs_weight": config.max_symbol_abs_weight,
    }
    assert config.book == "inverse_vol"


def test_trend_specification_screens_nothing(workspace: Workspace) -> None:
    # `TrendSignalNode` applies no eligibility screen; the declaration must say
    # so by carrying nothing but the resolved defaults.
    loaded = load_book_specification("trend", workspace=workspace)
    assert loaded.strategy.eligibility == Eligibility(**ELIGIBILITY_DEFAULTS)


def test_trend_specification_component_scores_identically(workspace: Workspace) -> None:
    symbols = list(TREND_V1_UNIVERSE)
    close, volume = _panel(symbols)
    book = BOOKS["trend"]
    params = BookParams()

    from_registry = book.build_signal(params, book.decision_every(params), None)
    from_specification = load_book_specification("trend", workspace=workspace).component("trend").component

    assert from_specification.horizon_bars == from_registry.horizon_bars
    assert from_specification.required_history == from_registry.required_history
    pd.testing.assert_series_equal(
        _scores(from_specification, close, volume),
        _scores(from_registry, close, volume),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# The CLI defaults the specification files transcribe
# ---------------------------------------------------------------------------


def test_cli_flag_defaults_are_the_values_the_specifications_declare() -> None:
    # The example files record `BookParams()`; `BookParams()` is only the CLI
    # default because `paper_loop.main` reads these flags into it. A changed
    # argparse default would silently make both files describe a book nobody
    # runs, so pin the flags themselves.
    args = _parse_args([])
    params = BookParams()
    assert args.mom_lookback == params.lookback_bars
    assert args.mom_skip == params.skip_bars
    assert args.decile == params.decile
    assert args.vol_ewma_bars == params.vol_ewma_bars
    assert args.max_gross == params.max_gross
    assert args.max_symbol_weight == params.max_symbol_abs_weight
    assert args.band == params.no_trade_band
    assert args.max_participation == params.max_participation
    assert args.min_notional == params.min_order_notional
    assert args.decision_every is params.decision_every is None
    assert (args.tif == "opg") == params.whole_shares
