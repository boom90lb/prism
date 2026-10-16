"""Parity of the extracted cost model (spec section 3.6, stage 2 "cost expression parity").

``execution/cost_model.py`` holds the former ``execution/target_weights.py::_cost_values``
body as :func:`prism.execution.cost_model.charge_weights`. The extraction is a **Parity**
change: the reproduction engine's charged cost must be bit-for-bit what it was before the
move, under all three declared spread modes (``flat``, ``bucket``, ``frame``).

Two independent guards, because they fail for different reasons:

1. :func:`test_charge_weights_matches_pre_extraction_arithmetic` drives ``charge_weights``
   and ``_reference_cost_values`` -- a verbatim transcription of the pre-extraction body,
   frozen in this file -- over a grid covering every branch of the function, and compares
   the raw IEEE-754 bytes of all eight returned values. Both run in the same interpreter,
   so this guard is environment-independent: a failure is a defect in the extraction.
2. :func:`test_backtest_target_weights_output_digests_unchanged` pins sha256 digests of
   the whole ``backtest_target_weights`` output, computed on the pre-extraction tree
   (commit 29a6c67) before the module existed. That guard also catches a change in the
   *call site*, which guard 1 cannot see.
"""

from __future__ import annotations

import hashlib
import itertools
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.cost_model import (
    SPREAD_BUCKET_SCHEDULE_V1,
    CostInputs,
    CostModel,
    CostStack,
    CostStackError,
    SpreadSpec,
    bucket_spread_bps,
    charge_weights,
)
from prism.execution.portfolio_state import state_from_shares
from prism.execution.target_weights import (
    PortfolioBacktestResult,
    backtest_target_weights,
)

# The eight keys of the reproduction cost row, in order. The order is part of the
# contract: ``backtest_target_weights`` builds its ``costs`` frame from these dicts and
# the column order is the recorded row shape.
_COST_KEYS = (
    "commission_spread",
    "impact",
    "borrow",
    "total",
    "turnover",
    "gross",
    "net",
    "dividend_return",
)


# fmt: off
# The transcription below is kept byte-identical to the pre-extraction source, so the
# formatter must not reflow it; that is the whole point of a frozen reference.
def _reference_cost_values(
    trade: np.ndarray,
    weights: np.ndarray,
    execution: ExecutionConfig,
    *,
    cost_multiplier: float,
    dividend_return: float,
    dollar_volume: np.ndarray | None = None,
    initial_capital: float = 1.0,
    spread_bps_per_name: np.ndarray | None = None,
) -> dict[str, float]:
    """Verbatim transcription of ``target_weights.py::_cost_values`` at commit 29a6c67.

    Do not refactor, tidy, or "improve" this body: it is the frozen pre-extraction
    arithmetic the moved code is measured against. If it ever needs to change, the
    parity claim it backs is void.
    """
    trade_abs = np.abs(trade)
    if spread_bps_per_name is None:
        commission_spread = float(
            trade_abs.sum()
            * (execution.commission_bps + execution.spread_bps)
            / 10_000.0
            * cost_multiplier
        )
    else:
        # Per-bucket spread (r2_design.md §3, I-9): commission stays scalar, the
        # spread is priced per name. None must reproduce the flat computation
        # bit-for-bit (frozen-v1 parity).
        commission_spread = float(
            (trade_abs * (execution.commission_bps + spread_bps_per_name)).sum()
            / 10_000.0
            * cost_multiplier
        )
    impact = float(
        np.square(trade_abs).sum()
        * execution.slippage_coeff
        / 10_000.0
        * cost_multiplier
    )
    if execution.adv_impact_coeff > 0.0 and dollar_volume is not None:
        valid = np.isfinite(dollar_volume)
        adv = np.where(valid, np.maximum(dollar_volume, execution.adv_floor_dollars), np.nan)
        participation = trade_abs * float(initial_capital) / adv
        if execution.adv_impact_model == "sqrt":
            participation = np.sqrt(participation)
        adv_bps = execution.adv_impact_coeff * participation
        impact += float(np.nansum(trade_abs * adv_bps / 10_000.0) * cost_multiplier)
    short_gross = float(np.abs(np.minimum(weights, 0.0)).sum())
    borrow = float(
        short_gross
        * execution.borrow_rate_bps_annual
        / 10_000.0
        / execution.trading_days_per_year
    )
    total = commission_spread + impact + borrow
    return {
        "commission_spread": commission_spread,
        "impact": impact,
        "borrow": borrow,
        "total": total,
        "turnover": float(trade_abs.sum()),
        "gross": float(np.abs(weights).sum()),
        "net": float(weights.sum()),
        "dividend_return": float(dividend_return),
    }
# fmt: on


def _bits(value: float) -> bytes:
    """Raw IEEE-754 bytes, so 0.0 and -0.0 and two NaN payloads are distinguishable."""
    return np.float64(value).tobytes()


def _assert_bitwise_equal(got: dict[str, float], want: dict[str, float], context: str) -> None:
    assert tuple(got) == _COST_KEYS, f"{context}: key order drifted -> {tuple(got)}"
    assert tuple(want) == _COST_KEYS
    for key in _COST_KEYS:
        assert _bits(got[key]) == _bits(want[key]), f"{context}: {key} differs bitwise: {got[key]!r} != {want[key]!r}"


# --------------------------------------------------------------------------------------
# The grid


def _weight_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, trade, holding) pairs, six names wide, covering the sign/zero cases."""
    rng = np.random.default_rng(20260907)
    long_short = rng.normal(0.0, 0.08, size=6)
    return [
        ("flat_book", np.zeros(6), np.zeros(6)),
        ("long_only", np.array([0.2, 0.1, 0.05, 0.0, 0.0, 0.0]), np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])),
        (
            "long_short",
            np.array([0.2, -0.1, 0.05, -0.25, 0.0, 0.3]),
            np.array([0.4, -0.3, 0.2, -0.1, 0.0, 0.15]),
        ),
        ("all_short", -np.abs(long_short), -np.abs(long_short) * 2.0),
        ("random", long_short, rng.normal(0.0, 0.15, size=6)),
        (
            "negative_zero",
            np.array([-0.0, 0.0, -0.0, 0.1, -0.1, 0.0]),
            np.array([-0.0, 0.0, 0.0, -0.0, 0.0, -0.0]),
        ),
        (
            "tiny",
            np.array([1e-12, -1e-12, 0.0, 5e-13, 0.0, -3e-14]),
            np.array([1e-9, -1e-9, 0.0, 0.0, 0.0, 0.0]),
        ),
    ]


def _dollar_volume_cases() -> list[tuple[str, np.ndarray | None]]:
    return [
        ("no_adv", None),
        ("adv_dense", np.array([9.0e8, 2.5e8, 4.0e7, 1.0e7, 3.0e6, 5.0e5])),
        # NaN ADV (unknown liquidity) and a sub-floor ADV that the pricer floors up.
        ("adv_with_nan", np.array([9.0e8, np.nan, 4.0e7, np.nan, 1.0e4, 0.0])),
        ("adv_all_nan", np.full(6, np.nan)),
        ("adv_zero", np.zeros(6)),
    ]


def _execution_cases() -> list[tuple[str, ExecutionConfig]]:
    return [
        ("reference", ExecutionConfig()),
        ("no_adv_term", ExecutionConfig(adv_impact_coeff=0.0)),
        ("adv_sqrt", ExecutionConfig(adv_impact_coeff=8.0, adv_impact_model="sqrt")),
        ("adv_linear", ExecutionConfig(adv_impact_coeff=8.0, adv_impact_model="linear")),
        (
            "wide_spread_no_borrow",
            ExecutionConfig(spread_bps=12.5, borrow_rate_bps_annual=0.0, slippage_coeff=0.0),
        ),
        (
            "zero_everything",
            ExecutionConfig(
                spread_bps=0.0,
                slippage_coeff=0.0,
                commission_bps=0.0,
                borrow_rate_bps_annual=0.0,
            ),
        ),
        (
            "coarse_year",
            ExecutionConfig(trading_days_per_year=360, borrow_rate_bps_annual=300.0, adv_floor_dollars=1.0e7),
        ),
    ]


def _spread_mode_cases(execution: ExecutionConfig, adv: np.ndarray | None) -> list[tuple[str, np.ndarray | None]]:
    """The three declared spread modes, as ``charge_weights`` sees them.

    ``flat`` is ``None`` (the scalar ``execution.spread_bps`` path); ``bucket`` is the
    per-name vector the schedule produces; ``frame`` is one bar's row of a day x name
    spread panel, which reaches the function in the same per-name-array shape but with
    values that vary by bar (``target_weights.py:369``).
    """
    symbols = [f"S{i}" for i in range(6)]
    liquidity = pd.Series(
        np.array([9.0e8, 2.5e8, 4.0e7, 1.0e7, np.nan, 0.0]) if adv is None else adv,
        index=symbols,
    )
    bucket = bucket_spread_bps(liquidity).to_numpy(dtype=float)
    panel = pd.DataFrame(
        [
            bucket,
            bucket * 1.5,
            np.full(6, float(execution.spread_bps)),
        ],
        columns=symbols,
    )
    frame_row = panel.to_numpy(dtype=float)[1]
    return [("flat", None), ("bucket", bucket), ("frame", frame_row)]


def test_charge_weights_matches_pre_extraction_arithmetic() -> None:
    """Bitwise parity over the full branch grid, all three spread modes."""
    multipliers = (0.0, 1.0, 2.5)
    dividends = (0.0, -0.0, 3.7e-4)
    capitals = (1.0, 1.0e6)
    cases = 0
    for (wlabel, trade, holding), (dlabel, adv), (elabel, execution) in itertools.product(
        _weight_cases(), _dollar_volume_cases(), _execution_cases()
    ):
        for slabel, spread in _spread_mode_cases(execution, adv):
            for multiplier, dividend, capital in zip(multipliers, dividends, capitals + (2.0e3,), strict=False):
                kwargs = dict(
                    cost_multiplier=multiplier,
                    dividend_return=dividend,
                    dollar_volume=adv,
                    initial_capital=capital,
                    spread_bps_per_name=spread,
                )
                got = charge_weights(trade, holding, execution, **kwargs)
                want = _reference_cost_values(trade, holding, execution, **kwargs)
                _assert_bitwise_equal(got, want, f"{wlabel}/{dlabel}/{elabel}/spread={slabel}/mult={multiplier}")
                cases += 1
    # Guard against a silently empty grid.
    assert cases == 7 * 5 * 7 * 3 * 3, cases


def test_charge_weights_defaults_match_the_explicit_call() -> None:
    """The spec gives ``cost_multiplier`` and ``dividend_return`` defaults; ``_cost_values``
    required them. The defaulted call must equal the explicit reference call."""
    trade = np.array([0.2, -0.1, 0.05, 0.0, 0.0, 0.0])
    holding = np.array([0.4, -0.3, 0.2, 0.0, 0.0, 0.0])
    execution = ExecutionConfig()
    got = charge_weights(trade, holding, execution)
    want = _reference_cost_values(trade, holding, execution, cost_multiplier=1.0, dividend_return=0.0)
    _assert_bitwise_equal(got, want, "defaults")


def test_cost_model_method_matches_the_module_function() -> None:
    """``CostModel.charge_weights`` is the spec's name for the same arithmetic."""
    trade = np.array([0.3, -0.2, 0.1, 0.0, -0.05, 0.0])
    holding = np.array([0.5, -0.4, 0.2, 0.1, -0.05, 0.0])
    execution = ExecutionConfig(adv_impact_coeff=8.0, adv_impact_model="sqrt")
    adv = np.array([9.0e8, np.nan, 4.0e7, 1.0e7, 3.0e6, 0.0])
    model = CostModel(execution)
    for spread in (None, np.array([1.0, 10.0, 5.0, 5.0, 10.0, 10.0])):
        got = model.charge_weights(
            trade,
            holding,
            cost_multiplier=1.25,
            dividend_return=1e-4,
            dollar_volume=adv,
            initial_capital=1.0e6,
            spread_bps_per_name=spread,
        )
        want = _reference_cost_values(
            trade,
            holding,
            execution,
            cost_multiplier=1.25,
            dividend_return=1e-4,
            dollar_volume=adv,
            initial_capital=1.0e6,
            spread_bps_per_name=spread,
        )
        _assert_bitwise_equal(got, want, f"CostModel spread={spread is not None}")


# --------------------------------------------------------------------------------------
# End-to-end: the reproduction engine's output is unchanged by the extraction


def _backtest_fixture() -> dict[str, object]:
    """A deterministic 40-bar, 6-name panel exercising every path the cost row touches.

    Built from a seeded generator with no I/O so the digests below are reproducible.
    """
    rng = np.random.default_rng(4242)
    n_bars, symbols = 40, [f"S{i}" for i in range(6)]
    index = pd.date_range("2024-01-02", periods=n_bars, freq="B")
    opens = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.012, size=(n_bars, len(symbols))), axis=0),
        index=index,
        columns=symbols,
    )
    raw = rng.normal(0.0, 0.18, size=(n_bars, len(symbols)))
    targets = pd.DataFrame(raw, index=index, columns=symbols)
    # A deliberate all-NaN drop row and a scattered-NaN row: both are cost-row paths.
    targets.iloc[7, :] = np.nan
    targets.iloc[13, 2] = np.nan
    dividends = pd.DataFrame(0.0, index=index, columns=symbols)
    dividends.iloc[9, 0] = 0.35
    dividends.iloc[22, 4] = 0.12
    dollar_volume = pd.DataFrame(
        np.abs(rng.lognormal(mean=17.0, sigma=1.6, size=(n_bars, len(symbols)))),
        index=index,
        columns=symbols,
    )
    dollar_volume.iloc[5, 1] = np.nan
    dollar_volume.iloc[11, 3] = 0.0
    liquidity = dollar_volume.median(axis=0, skipna=True)
    spread_series = bucket_spread_bps(liquidity)
    spread_frame = pd.DataFrame(
        np.tile(spread_series.to_numpy(dtype=float), (n_bars, 1)),
        index=index,
        columns=symbols,
    )
    # A per-fold re-bucket: the second half of the panel prices wider, and one cell is
    # missing so the flat fallback path is exercised too.
    spread_frame.iloc[20:, :] = spread_frame.iloc[20:, :].to_numpy() * 2.0
    spread_frame.iloc[3, 4] = np.nan
    return {
        "open_prices": opens,
        "target_weights": targets,
        "dividends": dividends,
        "dollar_volume": dollar_volume,
        "spread_series": spread_series,
        "spread_frame": spread_frame,
        "execution": ExecutionConfig(adv_impact_coeff=6.0, adv_impact_model="sqrt"),
    }


def _run_backtest(mode: str) -> PortfolioBacktestResult:
    fixture = _backtest_fixture()
    spread: pd.Series | pd.DataFrame | None = {
        "flat": None,
        "bucket": fixture["spread_series"],
        "frame": fixture["spread_frame"],
    }[mode]
    return backtest_target_weights(
        fixture["open_prices"],
        fixture["target_weights"],
        fixture["execution"],
        initial_capital=1.0e6,
        rebalance_band_weight=0.01,
        rebalance_cost_multiplier=1.25,
        max_gross_exposure=1.0,
        dividends=fixture["dividends"],
        dollar_volume=fixture["dollar_volume"],
        spread_bps_per_name=spread,
    )


def _digest(result: PortfolioBacktestResult) -> str:
    """sha256 over the raw float64 bytes of every numeric output, in a fixed order."""
    hasher = hashlib.sha256()
    for frame in (result.returns, result.equity):
        hasher.update(frame.to_numpy(dtype=float).tobytes())
    for frame in (result.target_weights, result.fill_weights, result.costs):
        hasher.update(",".join(map(str, frame.columns)).encode())
        hasher.update(frame.to_numpy(dtype=float).tobytes())
    for key in sorted(result.metrics):
        hasher.update(key.encode())
        hasher.update(np.float64(result.metrics[key]).tobytes())
    return hasher.hexdigest()


# Computed on the pre-extraction tree (commit 29a6c67, before cost_model.py existed) with
# numpy 2.x / pandas 2.x under .venv. A mismatch is either a behavioral change in the
# extraction -- a defect, this stage is Parity -- or a numeric change in the environment;
# the two are told apart by re-running the pinned commit, never by editing this table.
_PRE_EXTRACTION_DIGESTS = {
    "flat": "99cccd1dc17bb93a87b5fc4f8393c590cc0d88e1718dcd1d689810f9c75f4ba1",
    "bucket": "d002c1a1f5fe8a07dbc3f145449101070ea6e86d4c087e040877bb5ec5f7cf7b",
    "frame": "fa27f5e7bcd28ca622c51c62f9d5c9179a72b51d578196bf9250a7eda933e754",
}


@pytest.mark.parametrize("mode", ["flat", "bucket", "frame"])
def test_backtest_target_weights_output_digests_unchanged(mode: str) -> None:
    got = _digest(_run_backtest(mode))
    assert got == _PRE_EXTRACTION_DIGESTS[mode], (
        f"backtest_target_weights output changed in spread mode {mode!r}: "
        f"{got} != {_PRE_EXTRACTION_DIGESTS[mode]} "
        f"(numpy {np.__version__}, pandas {pd.__version__})"
    )


def test_flat_and_constant_frame_agree_end_to_end() -> None:
    """A constant frame at the flat value is bit-identical to the flat path, which is the
    property ``backtest_target_weights``' docstring promises and the extraction must keep."""
    fixture = _backtest_fixture()
    execution = ExecutionConfig(adv_impact_coeff=6.0, adv_impact_model="sqrt")
    constant = pd.DataFrame(
        float(execution.spread_bps),
        index=fixture["spread_frame"].index,
        columns=fixture["spread_frame"].columns,
    )
    common = dict(
        initial_capital=1.0e6,
        rebalance_cost_multiplier=1.25,
        dividends=fixture["dividends"],
        dollar_volume=fixture["dollar_volume"],
    )
    flat = backtest_target_weights(fixture["open_prices"], fixture["target_weights"], execution, **common)
    framed = backtest_target_weights(
        fixture["open_prices"],
        fixture["target_weights"],
        execution,
        spread_bps_per_name=constant,
        **common,
    )
    assert _digest(flat) == _digest(framed)


# --------------------------------------------------------------------------------------
# The bucket schedule


def test_bucket_spread_bps_matches_the_recorded_schedule() -> None:
    """Reproduces ``research/arbitrage/residual_walk_forward.py:77-87`` by value.

    Floors are inclusive lower bounds, steps descending; unknown liquidity (NaN, and any
    value below the last floor) takes the widest step -- the fail-safe direction.
    """
    liquidity = pd.Series(
        {
            "above_top": 1.0e9,
            "on_top_floor": 500e6,
            "second": 200e6,
            "on_second_floor": 100e6,
            "third": 50e6,
            "on_third_floor": 25e6,
            "widest": 1.0e6,
            "zero": 0.0,
            "negative": -1.0,
            "nan": np.nan,
        }
    )
    got = bucket_spread_bps(liquidity)
    assert list(got.index) == list(liquidity.index)
    assert got.dtype == np.float64
    expected = {
        "above_top": 1.0,
        "on_top_floor": 1.0,
        "second": 2.0,
        "on_second_floor": 2.0,
        "third": 5.0,
        "on_third_floor": 5.0,
        "widest": 10.0,
        "zero": 10.0,
        "negative": 10.0,
        "nan": 10.0,
    }
    assert got.to_dict() == expected


def test_bucket_spread_bps_accepts_a_declared_schedule() -> None:
    schedule = ((1.0e9, 0.5), (0.0, 20.0))
    got = bucket_spread_bps(pd.Series({"a": 2.0e9, "b": 5.0e8, "c": np.nan}), schedule=schedule)
    assert got.to_dict() == {"a": 0.5, "b": 20.0, "c": 20.0}


@pytest.mark.parametrize(
    "schedule",
    [
        (),
        ((0.0, 10.0), (500e6, 1.0)),  # ascending
        ((500e6, 1.0), (100e6, 2.0)),  # no 0.0 catch-all
        ((500e6, 1.0), (0.0, -1.0)),  # a negative charge
    ],
)
def test_bucket_spread_bps_refuses_a_malformed_schedule(schedule: tuple) -> None:
    with pytest.raises(ValueError):
        bucket_spread_bps(pd.Series({"a": 1.0}), schedule=schedule)


def test_bucket_spread_bps_refuses_a_non_numeric_liquidity_statistic() -> None:
    with pytest.raises(ValueError, match="numeric"):
        bucket_spread_bps(pd.Series({"a": "wide"}))


# --------------------------------------------------------------------------------------
# SpreadSpec: the three declared modes


def test_spread_spec_flat_carries_the_scalar_and_prices_no_vector() -> None:
    spec = SpreadSpec(mode="flat", flat_bps=1.0)
    assert spec.flat_bps == 1.0
    assert spec.per_name_bps(pd.Series({"a": 1.0e9})) is None


def test_spread_spec_bucket_prices_the_vector() -> None:
    spec = SpreadSpec(mode="bucket", schedule=SPREAD_BUCKET_SCHEDULE_V1)
    priced = spec.per_name_bps(pd.Series({"a": 1.0e9, "b": np.nan}))
    assert priced is not None
    assert priced.to_dict() == {"a": 1.0, "b": 10.0}


def test_spread_spec_frame_defers_to_the_supplied_panel() -> None:
    spec = SpreadSpec(mode="frame")
    with pytest.raises(ValueError, match="frame"):
        spec.per_name_bps(pd.Series({"a": 1.0e9}))


def test_spread_spec_frame_may_declare_its_fallbacks() -> None:
    """Section 3.6: a bar or name the panel does not cover falls back to a declared step,
    never to a silent zero. The frame spec carries that fallback declaration."""
    spec = SpreadSpec(mode="frame", flat_bps=1.0, schedule=SPREAD_BUCKET_SCHEDULE_V1)
    assert spec.flat_bps == 1.0
    assert spec.schedule == SPREAD_BUCKET_SCHEDULE_V1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "flat"},  # no flat_bps
        {"mode": "flat", "flat_bps": -1.0},
        {"mode": "bucket"},  # no schedule
        {"mode": "bucket", "schedule": ((0.0, 1.0), (500e6, 2.0))},
        {"mode": "flat", "flat_bps": 1.0, "schedule": SPREAD_BUCKET_SCHEDULE_V1},  # flat prices none
        {"mode": "widest"},
        {"mode": "bucket", "schedule": SPREAD_BUCKET_SCHEDULE_V1, "bound": "exclusive_lower"},
        {"mode": "bucket", "schedule": SPREAD_BUCKET_SCHEDULE_V1, "unknown_liquidity": "cheapest"},
    ],
)
def test_spread_spec_refuses_an_undeclarable_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SpreadSpec(**kwargs)


# --------------------------------------------------------------------------------------
# The dollar-space methods (stage 3): what they refuse, and where they agree


@pytest.mark.parametrize("method", ["charge_trade", "accrue_holding", "convex_terms"])
def test_the_dollar_methods_refuse_a_model_with_no_declared_stack(method: str) -> None:
    """Stage 2 deferred these to stage 3; stage 3 keeps the refusal where it still applies.

    A model built from an ``ExecutionConfig`` -- the reproduction engine's parameter object
    -- carries no ``cash_income_bps_annual``, ``financing_bps_annual``, ``short_proceeds``
    or ``holding_cost_settlement``. Charging dollars from it would either invent those four
    or charge zero for components nobody declared, and a zero that was never declared is
    indistinguishable in the record from a declared zero (spec section 3.6).
    """
    model = CostModel(ExecutionConfig())
    with pytest.raises(CostStackError, match="declared CostStack"):
        getattr(model, method)([])


def _stack(**overrides) -> CostStack:
    dollar_volume = {
        "spread_bucket": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
        },
        "participation": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
            "base": "current_equity",
        },
        "impact": {
            "estimator": "same_bar",
            "window": "trailing",
            "window_bars": 1,
            "refresh": "bar",
            "floor_dollars": 100_000.0,
        },
    }
    declared = {
        "spread": {"mode": "flat", "flat_bps": 2.0},
        "impact": {
            "quadratic_notional_coeff": 10.0,
            "participation_model": "sqrt",
            "participation_coeff": 5.0,
        },
        "dollar_volume": dollar_volume,
        "commission_bps": 1.0,
        "borrow_bps_annual": 50.0,
        "cash_income_bps_annual": 100.0,
        "financing_bps_annual": 200.0,
        "short_proceeds": "collateral_no_rebate",
        "holding_cost_settlement": "per_bar",
        "trading_days_per_year": 252,
    }
    declared.update(overrides)
    return CostStack.from_mapping(declared)


def test_the_loader_refuses_a_stack_that_omits_a_component() -> None:
    declared = _stack().declaration()
    declared.pop("borrow_bps_annual")
    with pytest.raises(CostStackError, match="omits"):
        CostStack.from_mapping(declared)


def test_the_loader_refuses_cash_income_above_financing() -> None:
    """The financing/income pair is convex only when ``c <= f``; without the check the DCP
    check fails inside the solver with no statement of why (spec section 3.6)."""
    with pytest.raises(CostStackError, match="convex"):
        _stack(cash_income_bps_annual=300.0, financing_bps_annual=200.0)


def test_the_loader_refuses_a_zero_impact_floor_and_accepts_the_other_two() -> None:
    """The impact denominator's floor is the only one that may not be zero.

    ``charge_weights`` computes ``participation = trade * capital / max(dollar_volume,
    adv_floor_dollars)`` (``cost_model.py`` around the ``np.maximum`` in the impact block),
    so a zero floor divides by zero on a name whose bar printed a price and no volume --
    a bar the synthetic scenario set deliberately writes -- and charges ``inf``.
    ``ExecutionConfig`` asserts the same thing, but an ``assert`` is stripped under
    ``python -O``, which is exactly when a silent ``inf`` in a charged cost would reach a
    real book; the stack refuses at load instead, in its own error class.

    ``spread_bucket`` and ``participation`` legitimately declare ``0.0``: neither divides
    by the statistic, and the shipped ``examples/reference/costs/retail_v1.toml`` declares
    ``0.0`` for both and ``100000.0`` for impact.
    """
    declared = _stack().declaration()
    declared["dollar_volume"]["impact"]["floor_dollars"] = 0.0
    with pytest.raises(CostStackError, match="floor_dollars must be > 0"):
        CostStack.from_mapping(declared)

    for use in ("spread_bucket", "participation"):
        declared = _stack().declaration()
        declared["dollar_volume"][use]["floor_dollars"] = 0.0
        assert CostStack.from_mapping(declared).dollar_volume[use].floor_dollars == 0.0


def test_the_adapter_is_a_rename_not_a_reinterpretation() -> None:
    execution = _stack().to_execution_config()
    assert execution.commission_bps == 1.0
    assert execution.spread_bps == 2.0
    assert execution.slippage_coeff == 10.0
    assert execution.adv_impact_coeff == 5.0
    assert execution.adv_impact_model == "sqrt"
    assert execution.adv_floor_dollars == 100_000.0
    assert execution.borrow_rate_bps_annual == 50.0
    assert execution.trading_days_per_year == 252


def test_a_reproduction_row_refuses_to_hide_a_financing_charge() -> None:
    """The reproduction record shape has three cost columns and no fourth to hide one in."""
    stack = _stack(borrow_bps_annual=0.0)
    model = CostModel(stack, CostInputs(as_of="2026-05-01"), equity=10_000.0, horizon_bars=1)
    state = state_from_shares("2026-05-01", {"AAA": 100.0}, {"AAA": 150.0}, cash=-5_000.0)
    accrual = model.accrue_holding(state, pd.Series({"AAA": 150.0}))
    assert accrual.financing > 0.0
    with pytest.raises(CostStackError, match="financing"):
        accrual.to_reproduction_row(turnover=0.0, gross=1.0, net=1.0, dividend_return=0.0)


# The four conditions under which the constructor's cost expression and the charged dollars
# are equal, stated in spec section 3.6 and part of the test rather than left to the reader:
#   1. every intent fills completely, in one fill;
#   2. every fill price equals the decision-close reference price;
#   3. prices are constant over the horizon_bars bars, so the marked state does not move;
#   4. equity is fixed at the decision-close mark for the whole horizon.
EQUITY = 1_000_000.0
HORIZON = 5
ADV = pd.Series({"AAA": 5.0e7, "BBB": 2.0e7, "CCC": 8.0e6})
TARGET_WEIGHTS = pd.Series({"AAA": 0.55, "BBB": 0.30, "CCC": -0.25})
PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 20.0})


def _model(stack: CostStack | None = None) -> CostModel:
    inputs = CostInputs(as_of="2026-05-01", adv_impact=ADV, adv_participation=ADV, adv_bucket=ADV)
    return CostModel(stack or _stack(), inputs, equity=EQUITY, horizon_bars=HORIZON)


def _full_fills(model: CostModel) -> tuple[list, dict[str, float]]:
    from prism.execution.portfolio_state import Fill

    fills, decided = [], {}
    for sid, weight in TARGET_WEIGHTS.items():
        qty = weight * EQUITY / float(PRICES[sid])
        client_order_id = f"2026-05-01:{sid}"
        decided[client_order_id] = abs(qty * float(PRICES[sid]))
        fills.append(
            Fill(
                execution_id=f"sim:{client_order_id}",
                client_order_id=client_order_id,
                sid=sid,
                ts="2026-05-04",
                qty=qty,
                fill_price=float(PRICES[sid]),
                reference_price=float(PRICES[sid]),
            )
        )
    return fills, decided


def test_cost_consistency_the_trade_side_in_dollars() -> None:
    """The constructor's one-time expression equals ``charge_trade``, in dollars."""
    model = _model()
    fills, decided = _full_fills(model)
    charged = model.charge_trade(fills, decided_notional=decided)
    terms = model.convex_terms(list(TARGET_WEIGHTS.index), adv_dollars=ADV)
    budgeted = terms.trade_costs(TARGET_WEIGHTS.to_numpy()) * EQUITY
    assert charged.trade_total == pytest.approx(budgeted, rel=1e-12)
    assert charged.impact_participation > 0.0, "the fixture must exercise the participation term"


def test_cost_consistency_the_accrual_side_in_dollars() -> None:
    """The constructor's holding expression equals ``accrue_holding`` summed over the horizon."""
    model = _model()
    shares = {sid: float(w) * EQUITY / float(PRICES[sid]) for sid, w in TARGET_WEIGHTS.items()}
    cash = EQUITY - sum(shares[sid] * float(PRICES[sid]) for sid in shares)
    state = state_from_shares("2026-05-04", shares, PRICES, cash=cash)
    assert state.equity(PRICES) == pytest.approx(EQUITY, rel=1e-12)
    per_bar = model.accrue_holding(state, PRICES)
    terms = model.convex_terms(list(TARGET_WEIGHTS.index), adv_dollars=ADV)
    budgeted = terms.holding_costs(TARGET_WEIGHTS.to_numpy()) * EQUITY
    assert per_bar.holding_total * HORIZON == pytest.approx(budgeted, rel=1e-12)


def test_cost_consistency_under_the_rebate_policy() -> None:
    """``proceeds_to_cash_with_rebate`` moves the same 500 into the income base on both sides."""
    stack = _stack(short_proceeds="proceeds_to_cash_with_rebate")
    model = _model(stack)
    shares = {sid: float(w) * EQUITY / float(PRICES[sid]) for sid, w in TARGET_WEIGHTS.items()}
    cash = EQUITY - sum(shares[sid] * float(PRICES[sid]) for sid in shares)
    state = state_from_shares("2026-05-04", shares, PRICES, cash=cash)
    per_bar = model.accrue_holding(state, PRICES)
    terms = model.convex_terms(list(TARGET_WEIGHTS.index), adv_dollars=ADV)
    assert per_bar.holding_total * HORIZON == pytest.approx(
        terms.holding_costs(TARGET_WEIGHTS.to_numpy()) * EQUITY, rel=1e-12
    )


def test_splitting_an_order_does_not_shrink_the_superlinear_charge() -> None:
    """The reason ``charge_trade`` takes one order's fills and the decided notional.

    Charged per fill, the quadratic term would fall by ``1/k`` and the sqrt participation
    term by ``k^-0.5`` across ``k`` equal fills. Charged on the decided order and allocated
    pro rata, the total is the same however the order is filled.
    """
    from prism.execution.portfolio_state import Fill

    model = _model()
    whole, decided = _full_fills(model)
    one_shot = model.charge_trade(whole, decided_notional=decided)
    split = []
    for fill in whole:
        for part in range(4):
            split.append(
                Fill(
                    execution_id=f"{fill.execution_id}:{part}",
                    client_order_id=fill.client_order_id,
                    sid=fill.sid,
                    ts=fill.ts,
                    qty=fill.qty / 4.0,
                    fill_price=fill.fill_price,
                    reference_price=fill.reference_price,
                )
            )
    in_pieces = model.charge_trade(split, decided_notional=decided)
    assert in_pieces.total == pytest.approx(one_shot.total, rel=1e-12)
    assert in_pieces.impact_quadratic == pytest.approx(one_shot.impact_quadratic, rel=1e-12)
    assert in_pieces.impact_participation == pytest.approx(one_shot.impact_participation, rel=1e-12)


def test_a_split_order_without_its_decided_notional_is_refused() -> None:
    """Never inferred from the fills: that is exactly the undercharge the rule prevents."""
    model = _model()
    whole, _ = _full_fills(model)
    halves = []
    from prism.execution.portfolio_state import Fill

    for fill in whole[:1]:
        for part in range(2):
            halves.append(
                Fill(
                    execution_id=f"{fill.execution_id}:{part}",
                    client_order_id=fill.client_order_id,
                    sid=fill.sid,
                    ts=fill.ts,
                    qty=fill.qty / 2.0,
                    fill_price=fill.fill_price,
                    reference_price=fill.reference_price,
                )
            )
    with pytest.raises(CostStackError, match="decided order notional"):
        model.charge_trade(halves)


def test_a_broker_fill_is_charged_its_commission_and_nothing_modeled() -> None:
    """No double charge in operation: the venue's price already carries spread and impact."""
    from prism.execution.portfolio_state import Fill

    model = _model()
    fill = Fill(
        execution_id="venue-1",
        client_order_id="2026-05-01:AAA",
        sid="AAA",
        ts="2026-05-04",
        qty=100.0,
        fill_price=100.25,
        reference_price=100.0,
        commission=0.35,
        source="broker",
    )
    charged = model.charge_trade([fill], decided_notional={"2026-05-01:AAA": 10_000.0})
    assert charged.commission == pytest.approx(0.35, abs=0.0)
    assert charged.spread == 0.0 and charged.impact_quadratic == 0.0 and charged.impact_participation == 0.0
    assert charged.total == pytest.approx(0.35, abs=0.0)


def test_a_fill_on_a_name_with_no_impact_statistic_is_refused_when_the_term_is_live() -> None:
    """The participation bound refuses the trade first, so a fill on such a name is a defect."""
    from prism.execution.portfolio_state import Fill

    inputs = CostInputs(as_of="2026-05-01", adv_impact=pd.Series({"AAA": float("nan")}))
    model = CostModel(_stack(), inputs, equity=EQUITY, horizon_bars=HORIZON)
    fill = Fill(
        execution_id="e1",
        client_order_id="2026-05-01:AAA",
        sid="AAA",
        ts="2026-05-04",
        qty=10.0,
        fill_price=100.0,
        reference_price=100.0,
    )
    with pytest.raises(CostStackError, match="accounting defect"):
        model.charge_trade([fill], decided_notional={"2026-05-01:AAA": 1_000.0})


def test_a_missing_spread_never_falls_back_to_zero() -> None:
    """Under ``bucket`` and ``frame`` an unknown name takes the widest step, not the cheapest."""
    stack = _stack(spread={"mode": "bucket", "schedule": [[100e6, 2.0], [0.0, 12.0]]})
    model = CostModel(stack, CostInputs(as_of="2026-05-01"), equity=EQUITY, horizon_bars=1)
    assert model.half_spread_bps("nobody-has-heard-of-this") == 12.0

    frame = pd.DataFrame({"AAA": [3.0]}, index=["2026-05-04"])
    framed = _stack(spread={"mode": "frame", "schedule": [[100e6, 2.0], [0.0, 12.0]]})
    model = CostModel(framed, CostInputs(as_of="2026-05-01", spread_bps=frame), equity=EQUITY, horizon_bars=1)
    assert model.half_spread_bps("AAA", "2026-05-04") == 3.0
    assert model.half_spread_bps("AAA", "2026-05-05") == 12.0
    assert model.half_spread_bps("BBB", "2026-05-04") == 12.0


# ---------------------------------------------------------------------------
# The shipped reference stacks
# ---------------------------------------------------------------------------

REFERENCE_COSTS = Path(__file__).resolve().parents[1] / "examples" / "reference" / "costs"

#: Content ids of the reference stacks that existed before retail_v3. retail_v1's id is
#: recorded in results/2026-09-*/comparison.json; a change here re-identifies every
#: record that cites it.
REFERENCE_STACK_IDS = {
    "retail_v1": "4cc998252c75a004496096bdf3ea62b573947d61cb310eb8d78055c58c46a8f0",
    "retail_v2": "357c89803e12d823cb96438b15aa35ae47ffcd2ff4e295ea08e054f1d2ec9a35",
}


def _reference_stack(name: str) -> CostStack:
    return CostStack.from_mapping(tomllib.loads((REFERENCE_COSTS / f"{name}.toml").read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", sorted(REFERENCE_STACK_IDS))
def test_the_existing_reference_stacks_keep_their_ids(name: str) -> None:
    assert _reference_stack(name).stack_id == REFERENCE_STACK_IDS[name]


def test_retail_v3_is_retail_v2_with_no_cash_income_and_a_square_root_impact_term() -> None:
    """Two declared changes and nothing else; the loader's convexity rule passes."""
    v2, v3 = _reference_stack("retail_v2"), _reference_stack("retail_v3")
    assert v3.cash_income_bps_annual == 0.0
    assert v3.cash_income_bps_annual <= v3.financing_bps_annual
    assert v3.impact.participation_model == "sqrt"
    # Y * sigma = 0.7 * 1.8% a day, in bp.
    assert v3.impact.participation_coeff == 126.0
    assert v3.dollar_volume["impact"].estimator == "same_bar"
    assert v3.dollar_volume["impact"].floor_dollars == 100_000.0

    differing = {
        key for key in v2.declaration() if key != "sources" and v2.declaration()[key] != v3.declaration()[key]
    }
    assert differing == {"cash_income_bps_annual", "impact"}
    impact_v2, impact_v3 = v2.declaration()["impact"], v3.declaration()["impact"]
    assert {key for key in impact_v2 if impact_v2[key] != impact_v3[key]} == {"participation_coeff"}
    assert v3.stack_id not in REFERENCE_STACK_IDS.values()

    sources = tomllib.loads((REFERENCE_COSTS / "retail_v3.toml").read_text(encoding="utf-8"))["source"]
    assert sources["impact"]["class"] == "assumption"
    assert "Almgren" in sources["impact"]["reference"] and "Toth" in sources["impact"]["reference"]


def test_the_canonical_engine_prices_an_active_participation_term_against_each_fill_sessions_volume() -> None:
    """``prism.evaluate._impact_dollar_volume``: what the engine binds as ``CostInputs.adv_impact``.

    Nothing for a stack whose term is inert, so retail_v1 and retail_v2 runs bind exactly the
    inputs they always did; close times volume keyed by bar for retail_v3, which the cost model
    reads at each fill's own bar; and a refusal for an estimator the engine does not implement.
    """
    from prism.evaluate import EvaluationError, _impact_dollar_volume  # noqa: PLC0415 - the engine's helper

    index = pd.DatetimeIndex(["2026-05-01", "2026-05-04"]).tz_localize("America/New_York")
    closes = pd.DataFrame({"AAA": [10.0, 11.0], "BBB": [20.0, np.nan]}, index=index)
    volumes = pd.DataFrame({"AAA": [1_000.0, 2_000.0], "BBB": [500.0, 700.0]}, index=index)
    for name in ("retail_v1", "retail_v2"):
        assert _impact_dollar_volume(_reference_stack(name), closes, volumes) is None
    v3 = _reference_stack("retail_v3")
    panel = _impact_dollar_volume(v3, closes, volumes)
    assert list(panel.index) == ["2026-05-01", "2026-05-04"]
    assert panel.loc["2026-05-04", "AAA"] == 22_000.0 and np.isnan(panel.loc["2026-05-04", "BBB"])
    model = CostModel(v3, CostInputs(as_of="2026-05-04", adv_impact=panel), equity=100_000.0)
    # Below the $100k floor the floor prices it; a fill on a bar with no statistic is refused.
    assert model._adv_impact("AAA", "2026-05-04") == 100_000.0
    with pytest.raises(CostStackError, match="dollar_volume.impact"):
        model._adv_impact("BBB", "2026-05-04")

    declared = v3.declaration()
    declared["dollar_volume"]["impact"] = dict(declared["dollar_volume"]["impact"], estimator="median",
                                               window_bars=20, refresh="decision")
    with pytest.raises(EvaluationError, match="same_bar"):
        _impact_dollar_volume(CostStack.from_mapping(declared), closes, volumes)
