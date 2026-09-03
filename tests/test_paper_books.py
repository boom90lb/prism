"""scripts/ paper-loop book registry (CLI assembly, offline).

The registry exists so the CLI shell holds no logic worth testing — which
makes THESE the tests that logic never had. The load-bearing pin: the
momentum spec under CLI defaults is field-identical to the certified B1
paper config (G6, docs/risk_profile_schema.md) — a drifted default here is a
silent fork of the promotion instrument.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prism.io.universe_file import load_universe_symbols
from prism.live.risk_profile import (
    CERTIFIED_B1_PAPER_CONFIG,
    assert_research_paper_bit_identity,
)
from prism.scripts.paper_books import (
    BOOKS,
    BookParams,
    apply_certified_paper_pin,
    build_safety_config,
    resolve_cli_universe,
    valuation_masked_membership,
)
from prism.signal.trend_node import TREND_V1_UNIVERSE

# ---------------------------------------------------------------------------
# G6: CLI defaults == the certified paper instrument
# ---------------------------------------------------------------------------


def test_momentum_defaults_reproduce_certified_paper_config() -> None:
    spec = BOOKS["momentum"]
    params = BookParams()
    config = spec.build_config(params, spec.decision_every(params))
    assert_research_paper_bit_identity(config)  # raises on any field drift


def test_certified_pin_is_idempotent_on_default_params() -> None:
    params = apply_certified_paper_pin(BookParams(), CERTIFIED_B1_PAPER_CONFIG)
    spec = BOOKS["momentum"]
    config = spec.build_config(params, spec.decision_every(params))
    assert_research_paper_bit_identity(config)


def test_certified_pin_overrides_operator_drift() -> None:
    drifted = BookParams(decile=0.25, max_gross=2.0, decision_every=5, min_order_notional=50.0)
    pinned = apply_certified_paper_pin(drifted, CERTIFIED_B1_PAPER_CONFIG)
    spec = BOOKS["momentum"]
    config = spec.build_config(pinned, spec.decision_every(pinned))
    assert_research_paper_bit_identity(config)


# ---------------------------------------------------------------------------
# Per-book spec facts
# ---------------------------------------------------------------------------


def test_registry_prefixes_and_defaults() -> None:
    assert BOOKS["momentum"].order_id_prefix == "mom:"
    assert BOOKS["trend"].order_id_prefix == "trd:"
    assert BOOKS["ensemble"].order_id_prefix == ""
    assert BOOKS["momentum"].default_decision_every == 21
    assert BOOKS["trend"].default_decision_every == 21
    assert BOOKS["ensemble"].default_decision_every == 1
    assert BOOKS["momentum"].default_max_missing == 0.10
    assert BOOKS["ensemble"].default_max_missing == 0.0


def test_trend_config_participation_defaults_on_but_yields_to_cli() -> None:
    spec = BOOKS["trend"]
    default = spec.build_config(BookParams(), spec.decision_every(BookParams()))
    assert default.book == "inverse_vol"
    assert default.max_participation == 0.05
    explicit = spec.build_config(BookParams(max_participation=0.02), 21)
    assert explicit.max_participation == 0.02


def test_trend_default_universe_is_the_pinned_list() -> None:
    assert BOOKS["trend"].default_universe == tuple(TREND_V1_UNIVERSE)


def test_ensemble_threads_decision_every_through() -> None:
    # Pre-registry the CLI silently dropped --decision-every for this book.
    config = BOOKS["ensemble"].build_config(BookParams(decision_every=5), 5)
    assert config.decision_every == 5
    assert config.book == "directional"


def test_momentum_signal_horizon_tracks_cadence() -> None:
    spec = BOOKS["momentum"]
    signal = spec.build_signal(BookParams(), 21, None)
    assert signal.horizon_bars == 21


def test_decision_every_override_beats_spec_default() -> None:
    assert BOOKS["momentum"].decision_every(BookParams(decision_every=5)) == 5
    assert BOOKS["momentum"].decision_every(BookParams()) == 21


# ---------------------------------------------------------------------------
# Universe resolution and the shared file parser
# ---------------------------------------------------------------------------


def test_universe_precedence_file_beats_symbols(tmp_path) -> None:
    path = tmp_path / "u.txt"
    path.write_text("# comment\naapl\n\nMSFT\n", encoding="utf-8")
    assert resolve_cli_universe(path, "GOOG", BOOKS["momentum"]) == ["AAPL", "MSFT"]


def test_universe_symbols_flag_parsed_and_uppercased() -> None:
    assert resolve_cli_universe(None, " aapl, msft ,", BOOKS["momentum"]) == ["AAPL", "MSFT"]


def test_universe_trend_falls_back_to_pinned_list() -> None:
    assert resolve_cli_universe(None, None, BOOKS["trend"]) == list(TREND_V1_UNIVERSE)


def test_universe_missing_everything_exits() -> None:
    with pytest.raises(SystemExit, match="provide --symbols or --universe-file"):
        resolve_cli_universe(None, None, BOOKS["ensemble"])


def test_load_universe_symbols_empty_file_raises(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no symbols"):
        load_universe_symbols(path)
    assert load_universe_symbols(path, require_nonempty=False) == []


# ---------------------------------------------------------------------------
# Membership mask and safety derivation
# ---------------------------------------------------------------------------


def test_valuation_mask_none_without_extras_true_on_score_columns() -> None:
    idx = pd.bdate_range("2026-06-01", periods=3)
    close = pd.DataFrame(100.0, index=idx, columns=["AAA", "BBB", "POOL"])
    assert valuation_masked_membership(close, ["AAA", "BBB"], has_extras=False) is None
    mask = valuation_masked_membership(close, ["AAA", "BBB"], has_extras=True)
    assert mask is not None
    assert bool(mask["AAA"].all()) and bool(mask["BBB"].all())
    assert not bool(mask["POOL"].any())  # valuation-only extra: never rankable


def test_safety_defaults_derive_from_universe_and_cap(tmp_path) -> None:
    safety = build_safety_config(
        run_dir=tmp_path,
        kill_switch_arg=None,
        max_drawdown=0.5,
        max_order_fraction=None,
        max_symbol_abs_weight=0.10,
        max_orders=None,
        n_symbols=500,
    )
    assert safety.kill_switch == tmp_path / "KILL_SWITCH"
    assert safety.max_drawdown == 0.5
    assert safety.max_order_fraction == pytest.approx(0.20)  # 2x the symbol cap
    assert safety.max_orders == 2 * 500 + 10


def test_safety_explicit_off_switches(tmp_path) -> None:
    from pathlib import Path

    safety = build_safety_config(
        run_dir=tmp_path,
        kill_switch_arg=Path("off"),
        max_drawdown=0.0,
        max_order_fraction=0.0,
        max_symbol_abs_weight=0.10,
        max_orders=0,
        n_symbols=10,
    )
    assert safety.kill_switch is None
    assert safety.max_drawdown is None
    assert safety.max_order_fraction is None
    assert safety.max_orders is None
