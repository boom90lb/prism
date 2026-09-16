"""The stage 2 reproduction of the recorded B1 momentum record (spec section 5).

Three kinds of check live here, and only the third needs the untracked bar
caches.

*The specification is the recorded configuration.* ``examples/momentum/demotion_b1.toml``
is asserted field by field against ``results/demotion_b1/config.json``, which is
tracked. A value that drifts on either side fails here rather than quietly
reproducing a different configuration under the recorded name.

*The construction is the recorded driver's.* The band-then-cap-then-bound step,
the bucket schedule applied to a formation window, the charged-spread cadence
and the due-ness predicate are checked against small hand-built cases whose
expected values are written out independently of the code under test.

*The golden reproduction.* The evaluator's own fold-0 metrics are compared with
the fold-0 metrics ``results/demotion_b1/folds.json`` records, at the tolerance
declared in ``FOLD_TOLERANCE`` below. It skips when the reproduction dataset is
absent, because the vendor bar caches are not redistributable and are not in
the repository; the command that builds it is in the skip message.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.composition import ComponentPlan, ComponentState
from prism.config import ExecutionConfig
from prism.evaluate import (
    _apply_construction,
    _bucket_spread,
    _charged_spread,
    _Construction,
    _due,
    _FoldCostInputs,
    _trailing_dollar_volume,
    walk_forward_evaluate,
)
from prism.execution.cost_model import bucket_spread_bps
from prism.io.dataset import Dataset
from prism.io.legacy_import import (
    LegacyImportError,
    _membership_rows,
    import_legacy,
    round_trip_panel,
    session_index,
    sid_by_symbol,
)
from prism.portfolio.construct import cap_book, closed_form_band, step_no_trade_band
from prism.scripts.reproduce import ARMS, FOLD_TABLE, FOLD_TABLE_SHA256, RECORDED_HEADLINE
from prism.signal.momentum_node import MomentumSignalNode
from prism.spec import load_spec
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace

REPO = Path(__file__).resolve().parents[1]
#: The recorded record, read from THIS checkout. Its five JSON artifacts
#: (``config.json``, ``folds.json``, ``summary.json``, ``claim_packet.json``,
#: ``breadth_diagnostic.json``) are tracked, so every assertion below runs
#: wherever the repository is cloned -- a CI runner included. An absolute path
#: to one operator's working copy would have made these tests skip everywhere
#: else and pass green while checking nothing. The four CSV time series in the
#: same recorded directory are untracked by design
#: (``docs/certifications/001-residual-reversion-daily-negative.md``: local and
#: regenerable) and nothing here reads them.
RECORDED = REPO / "results" / "demotion_b1"
DATASET = REPO / "data" / "frozen" / "demotion_b1"

#: The declared tolerance for the golden reproduction, and the reason for the
#: number. The evaluator and the recorded driver sum the same per-bar cost and
#: return terms in a different order over the same 574-name cross-section, so
#: the two differ by floating-point association and by nothing else that is
#: intended. Double-precision association error over ~10^3 bars x ~10^2 held
#: names accumulates at the 1e-13 level relative; 1e-9 relative is four orders
#: of magnitude of headroom over that and still four orders below the smallest
#: economically meaningful difference on these figures (the participation
#: bound, the smallest knob in the recorded configuration, moves total return
#: by 1e-3 relative). A difference above this is an engine difference and is
#: reported as a finding, never absorbed.
FOLD_TOLERANCE = 1e-9

#: The last session the recorded sample covers, and the number of priced cells
#: the 574-name panel holds over ``start_date``..that session. Both are Run
#: figures from the real caches, stated here so a silent change in either --
#: a truncated import, a dropped symbol -- is a failure and not a quieter pass.
RECORDED_LAST_SESSION = "2026-06-15"
RECORDED_PANEL_CELLS = 905_669


def _recorded_config() -> dict:
    # Not a skip: config.json is tracked, so its absence is a damaged checkout
    # and not an environment this suite may quietly pass in.
    path = RECORDED / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"the tracked recorded record {path} is missing from this checkout")
    return json.loads(path.read_text())


def _recorded_folds() -> list[dict]:
    return json.loads((RECORDED / "folds.json").read_text())


@pytest.fixture(scope="module")
def workspace() -> Workspace:
    return Workspace(root=REPO)


@pytest.fixture(scope="module")
def spec(workspace: Workspace):
    return load_spec(REPO / ARMS["bucket"], workspace=workspace)


# ---------------------------------------------------------------------------
# The specification is the recorded configuration
# ---------------------------------------------------------------------------


def test_the_specification_restates_the_recorded_walk_and_signal_blocks(spec) -> None:
    config = _recorded_config()
    walk = config["walk"]
    signal = config["signal"]
    assert walk["sleeve_mode"] == "momentum_only", "this reproduction target is the momentum-only sleeve"

    component = spec.components[0].spec
    assert component.params["lookback_bars"] == walk["mom_lookback_bars"]
    assert component.params["skip_bars"] == walk["mom_skip_bars"]
    assert component.update_every_bars == walk["mom_decision_every"] == signal["decision_every"]
    assert component.valid_for_bars == walk["mom_decision_every"]
    assert component.to_weights == "decile_long_short"
    assert component.to_weights_params["decile"] == pytest.approx(walk["mom_decile"])
    assert component.to_weights_params["max_gross"] == pytest.approx(walk["max_gross"])
    assert component.to_weights_params["max_symbol_abs_weight"] == pytest.approx(walk["max_symbol_abs_weight"])


def test_the_specification_restates_the_recorded_construction_block(spec) -> None:
    config = _recorded_config()
    walk = config["walk"]
    construction = _Construction.from_spec(spec)
    assert construction.band_mode == walk["band_mode"] == "closed_form"
    assert construction.gamma_risk == pytest.approx(config["band_gamma_risk"])
    assert construction.max_participation == pytest.approx(walk["max_participation"])
    assert construction.spread_mode == walk["spread_mode"] == "bucket"
    assert construction.max_gross == pytest.approx(walk["max_gross"])
    assert construction.max_symbol_abs_weight == pytest.approx(walk["max_symbol_abs_weight"])
    # The recorded driver bands, then re-caps, then bounds.
    assert construction.step_order == "construct_band_cap_participation"
    # The bound's statistic is the one the eligibility screen uses.
    assert construction.adv_window_bars == config["signal"]["dollar_volume_window"]
    assert construction.adv_statistic == "median"
    assert construction.adv_floor_dollars == 0.0


def test_the_specification_restates_the_recorded_eligibility_screen(spec) -> None:
    signal = _recorded_config()["signal"]
    screen = spec.strategy.eligibility
    assert screen.min_price == pytest.approx(signal["min_price"])
    assert screen.min_median_dollar_volume == pytest.approx(signal["min_median_dollar_volume"])
    assert screen.dollar_volume_window == signal["dollar_volume_window"]
    # The full-history gate is corr_window bars of finite trailing returns.
    assert screen.return_history_bars == signal["corr_window"]
    assert screen.membership_required is True
    assert screen.exclude_asset_classes == ()


def test_the_flat_arm_differs_from_the_recorded_arm_only_in_the_spread_mode(workspace: Workspace) -> None:
    """The two arms are one knob apart, and the knob is the one the exit names."""
    bucket = load_spec(REPO / ARMS["bucket"], workspace=workspace)
    flat = load_spec(REPO / ARMS["flat"], workspace=workspace)
    assert _Construction.from_spec(bucket).spread_mode == "bucket"
    assert _Construction.from_spec(flat).spread_mode == "flat"
    left = bucket.resolved()
    right = flat.resolved()
    left["strategy"]["name"] = right["strategy"]["name"] = "same"
    assert left["construction"].pop("spread") == {"mode": "bucket"}
    assert right["construction"].pop("spread") == {"mode": "flat"}
    assert left == right
    # Two configurations are two identities: the reproduction never charges one
    # arm's costs against the other's record.
    assert bucket.config_id() != flat.config_id()


# ---------------------------------------------------------------------------
# The fold table is the recorded geometry
# ---------------------------------------------------------------------------


def test_the_imported_fold_table_is_the_recorded_geometry() -> None:
    from_file = FoldTable.from_toml(REPO / FOLD_TABLE)
    assert from_file.content_hash == FOLD_TABLE_SHA256
    assert len(from_file.folds) == 21

    imported = FoldTable.from_json(
        RECORDED / "folds.json", table_id=from_file.table_id, source=from_file.source, timezone=from_file.timezone
    )
    assert imported.content_hash == from_file.content_hash
    imported.assert_matches(from_file)

    recorded = _recorded_folds()
    assert [row["formation_rows"] for row in recorded] == [312] * 21
    assert [row["test_rows"] for row in recorded][:-1] == [63] * 20
    assert recorded[-1]["test_rows"] == 48, "the last test window is truncated by the end of the sample"


# ---------------------------------------------------------------------------
# The construction is the recorded driver's
# ---------------------------------------------------------------------------


def _construction(**overrides) -> _Construction:
    base = dict(
        step_order="construct_band_cap_participation",
        max_gross=1.0,
        max_symbol_abs_weight=0.35,
        band_mode="closed_form",
        band_width=0.0,
        gamma_risk=1.0,
        max_participation=None,
        adv_window_bars=20,
        adv_statistic="median",
        adv_floor_dollars=0.0,
        participation_base="initial_capital",
        spread_mode="bucket",
    )
    base.update(overrides)
    return _Construction(**base)


def test_the_construction_step_is_band_then_cap_then_bound() -> None:
    """The three steps, in order, against a hand-computed expectation.

    ``held`` is a book of two names; ``raw`` moves one past the band and one
    inside it; the blend then exceeds ``max_gross``, so the gross scale-down
    bites and the per-name clip bites after it; and the bound is loose enough
    not to. Every expected number below is written out from the rule, not read
    off the implementation.
    """
    index = pd.Index(["A", "B"])
    held = pd.Series([0.6, 0.4], index=index)
    raw = pd.Series([0.9, 0.41], index=index)
    band = pd.Series([0.05, 0.05], index=index)
    adv = pd.Series([1e12, 1e12], index=index)

    out = _apply_construction(
        raw,
        held,
        _construction(max_participation=0.05, max_gross=1.0),
        band=band,
        dollar_volume=adv,
        capital=1.0,
    )
    # A moves (|0.9 - 0.6| = 0.30 > 0.05); B does not (|0.41 - 0.40| = 0.01).
    # The blend is 0.90 + 0.40 = 1.30 gross, scaled down by 1/1.3 to
    # (0.6923, 0.3077), and A is then clipped to max_symbol_abs_weight = 0.35.
    # The clip is down-only, so the row ends below max_gross rather than being
    # scaled back up: cap_book never grosses a book up.
    assert out["A"] == pytest.approx(0.35)
    assert out["B"] == pytest.approx(0.4 / 1.3)
    assert out.abs().sum() == pytest.approx(0.35 + 0.4 / 1.3)


def test_the_construction_step_matches_the_recorded_drivers_online_band() -> None:
    """Parity with ``_online_banded_targets`` (research/arbitrage/residual_walk_forward.py:230-258).

    The reference below is the recorded driver's loop body, transcribed here so
    the comparison is against what that file does rather than against a second
    reading of this one.
    """
    rng = np.random.default_rng(11)
    names = [f"S{i}" for i in range(12)]
    frame = pd.DataFrame(rng.normal(scale=0.08, size=(9, 12)), index=range(9), columns=names)
    band = pd.Series(rng.uniform(0.0, 0.05, size=12), index=names)
    construction = _construction(max_participation=None)

    reference_held = pd.Series(0.0, index=pd.Index(names))
    reference: list[np.ndarray] = []
    for day in frame.index:
        stepped = step_no_trade_band(reference_held, frame.loc[day], band)
        capped = cap_book(stepped.to_frame().T, construction.max_gross, construction.max_symbol_abs_weight).iloc[0]
        reference_held = capped.reindex(pd.Index(names)).fillna(0.0)
        reference.append(reference_held.to_numpy(dtype=float))

    held = pd.Series(0.0, index=pd.Index(names))
    produced: list[np.ndarray] = []
    for day in frame.index:
        held = _apply_construction(frame.loc[day], held, construction, band=band, dollar_volume=None, capital=1.0)
        produced.append(held.to_numpy(dtype=float))

    np.testing.assert_allclose(np.vstack(produced), np.vstack(reference), rtol=0.0, atol=0.0)


def test_a_declared_participation_bound_with_no_statistic_is_refused_not_waived() -> None:
    index = pd.Index(["A"])
    with pytest.raises(Exception, match="refused, never waived"):
        _apply_construction(
            pd.Series([0.5], index=index),
            pd.Series([0.0], index=index),
            _construction(max_participation=0.05),
            band=0.0,
            dollar_volume=None,
            capital=1.0,
        )


def test_the_closed_form_band_is_sized_on_the_folds_own_spread_buckets() -> None:
    """The band's cost input is per-name when the buckets are, which changes the width.

    A name in the widest bucket carries a round-trip cost ten times the
    cheapest name's and therefore a band ``10 ** (1/3)`` times wider at the
    same target-change variance. Pricing every name at the flat spread would
    understate the illiquid names' band, which is the direction that trades too
    often.
    """
    sigma2 = np.array([4e-7, 4e-7])
    flat = closed_form_band(sigma2, 2.0 * (1.0 + 1.0) / 10_000.0, gamma_risk=1.0)
    bucketed = closed_form_band(sigma2, 2.0 * (1.0 + np.array([1.0, 10.0])) / 10_000.0, gamma_risk=1.0)
    assert bucketed[0] == pytest.approx(flat[0])
    assert bucketed[1] > flat[1]
    assert bucketed[1] / bucketed[0] == pytest.approx((11.0 / 2.0) ** (1.0 / 3.0))


def test_the_bucket_spread_prices_a_formation_window_and_never_cheapens_unknown_liquidity() -> None:
    index = pd.date_range("2026-01-05", periods=4, freq="B", tz="America/New_York")
    dollar = pd.DataFrame(
        {
            "LIQUID": [6e8, 6e8, 6e8, 6e8],
            "MID": [2e8, 2e8, 2e8, 2e8],
            "THIN": [1e7, 1e7, 1e7, 1e7],
            "UNKNOWN": [np.nan] * 4,
        },
        index=index,
    )
    spread = _bucket_spread(dollar, "median")
    assert spread["LIQUID"] == 1.0
    assert spread["MID"] == 2.0
    assert spread["THIN"] == 10.0
    # An all-NaN window is unknown liquidity, which lands in the widest step.
    assert spread["UNKNOWN"] == 10.0
    pd.testing.assert_series_equal(spread, bucket_spread_bps(dollar.median(axis=0, skipna=True)))


def test_the_trailing_dollar_volume_statistic_refuses_a_partial_window() -> None:
    index = pd.date_range("2026-01-05", periods=5, freq="B", tz="America/New_York")
    close = pd.DataFrame({"A": [10.0] * 5, "B": [10.0] * 5}, index=index)
    volume = pd.DataFrame({"A": [100.0] * 5, "B": [100.0, np.nan, 100.0, 100.0, 100.0]}, index=index)
    stat = _trailing_dollar_volume(close, volume, window=4, statistic="median")
    assert stat["A"] == pytest.approx(1000.0)
    assert math.isnan(stat["B"]), "a name without a full window has no statistic, and the bound refuses it"


# ---------------------------------------------------------------------------
# The charged spread follows the declared refresh cadence
# ---------------------------------------------------------------------------


class _FakeResolved:
    def __init__(self, index: pd.DatetimeIndex, universe: tuple[str, ...]) -> None:
        self.session_index = index
        self.session_labels = tuple(str(ts.date()) for ts in index)
        self.universe = universe


class _FakeFold:
    def __init__(self, fold: int, test: tuple[str, str]) -> None:
        self.fold = fold
        self.test = test


def _spread_inputs(index: pd.DatetimeIndex, universe: tuple[str, ...]):
    return (
        _FoldCostInputs(0, ("a", "b"), pd.Series([1.0, 2.0], index=pd.Index(universe)), 0.0),
        _FoldCostInputs(1, ("c", "d"), pd.Series([5.0, 10.0], index=pd.Index(universe)), 0.0),
    )


def test_cost_refresh_once_prices_the_whole_sample_from_the_first_formation_window() -> None:
    index = pd.date_range("2026-01-05", periods=6, freq="B", tz="America/New_York")
    universe = ("A", "B")
    resolved = _FakeResolved(index, universe)
    folds = [
        _FakeFold(0, (resolved.session_labels[0], resolved.session_labels[2])),
        _FakeFold(1, (resolved.session_labels[3], resolved.session_labels[5])),
    ]
    charged = _charged_spread(resolved, folds=folds, cost_inputs=_spread_inputs(index, universe), cost_refresh="once")
    assert isinstance(charged, pd.Series)
    assert list(charged) == [1.0, 2.0]


def test_cost_refresh_fold_prices_each_folds_test_bars_and_the_bar_its_last_decision_fills_on() -> None:
    index = pd.date_range("2026-01-05", periods=6, freq="B", tz="America/New_York")
    universe = ("A", "B")
    resolved = _FakeResolved(index, universe)
    folds = [
        _FakeFold(0, (resolved.session_labels[0], resolved.session_labels[2])),
        _FakeFold(1, (resolved.session_labels[3], resolved.session_labels[5])),
    ]
    charged = _charged_spread(resolved, folds=folds, cost_inputs=_spread_inputs(index, universe), cost_refresh="fold")
    assert isinstance(charged, pd.DataFrame)
    # Fold 0's own buckets reach bar 3, which is where its last decision fills.
    assert list(charged.iloc[0]) == [1.0, 2.0]
    assert list(charged.iloc[2]) == [1.0, 2.0]
    # Fold 1 then overwrites its own window.
    assert list(charged.iloc[3]) == [5.0, 10.0]
    assert list(charged.iloc[5]) == [5.0, 10.0]


def test_no_declared_schedule_is_the_flat_scalar_and_is_bit_identical_to_none() -> None:
    index = pd.date_range("2026-01-05", periods=3, freq="B", tz="America/New_York")
    resolved = _FakeResolved(index, ("A",))
    inputs = (_FoldCostInputs(0, ("a", "b"), None, 0.0),)
    folds = [_FakeFold(0, (resolved.session_labels[0], resolved.session_labels[2]))]
    assert _charged_spread(resolved, folds=folds, cost_inputs=inputs, cost_refresh="once") is None
    assert _charged_spread(resolved, folds=folds, cost_inputs=inputs, cost_refresh="fold") is None


# ---------------------------------------------------------------------------
# The non-decision fast path decides nothing
# ---------------------------------------------------------------------------


def test_due_ness_matches_the_composition_state_machine() -> None:
    """``_due`` skips a panel read only where composition would score nothing.

    The predicate is duplicated from ``_compose_one`` so the loop can avoid the
    read, so it is checked against the same three cases that function branches
    on: never scored, refreshed within the cadence, and past the cadence.
    """
    plan = ComponentPlan(
        component=MomentumSignalNode(),
        component_id="momentum",
        update_every_bars=21,
        valid_for_bars=21,
        to_weights="decile_long_short",
        to_weights_params={"decile": 0.1},
    )
    labels = [f"2026-{1 + i // 20:02d}-{1 + i % 20:02d}" for i in range(60)]
    positions = {label: number for number, label in enumerate(labels)}

    assert _due([plan], {}, positions, 0) is True, "a component that has never scored is due"
    state = {"momentum": ComponentState(component_id="momentum", last_output_bar=labels[0])}
    assert _due([plan], state, positions, 1) is False
    assert _due([plan], state, positions, 20) is False
    assert _due([plan], state, positions, 21) is True, "elapsed == update_every_bars is due"
    # A refresh the calendar cannot place is due rather than silently carried.
    orphan = {"momentum": ComponentState(component_id="momentum", last_output_bar="1999-01-01")}
    assert _due([plan], orphan, positions, 3) is True


# ---------------------------------------------------------------------------
# The legacy import's translations
# ---------------------------------------------------------------------------


def _membership_frame(rows: list[tuple[str, str, str]]) -> Path:
    frame = pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "start": pd.to_datetime([r[1] for r in rows]),
            "end": pd.to_datetime([r[2] for r in rows]),
        }
    )
    return frame


def test_membership_translation_floors_unknown_starts_and_drops_out_of_sample_intervals(tmp_path: Path) -> None:
    path = tmp_path / "membership.parquet"
    _membership_frame(
        [
            ("AAA", "1990-01-01", "2026-06-16"),  # opened before the sample
            ("BBB", "2022-03-01", "2024-05-05"),  # wholly inside
            ("CCC", "1990-01-01", "2013-04-04"),  # closed before the sample
            ("DDD", "2027-01-04", "2027-12-31"),  # opens after the sample
        ]
    ).to_parquet(path, index=False)
    rows, intervals = _membership_rows(
        path,
        ["AAA", "BBB", "CCC", "DDD"],
        index_id="sp500",
        first_session=dt.date(2020, 1, 2),
        last_session=dt.date(2026, 6, 15),
    )
    assert intervals == 4, "the source's own interval count is reported, not the written one"
    written = {row["sid"]: row for row in rows.to_dict("records")}
    assert set(written) == {"AAA", "BBB"}
    assert written["AAA"]["valid_from"] is None and written["AAA"]["floor_unknown"] is True
    assert written["BBB"]["valid_from"] == "2022-03-01" and written["BBB"]["floor_unknown"] is False
    assert written["AAA"]["valid_to"] == "2026-06-16", "an as-of end is not opened at stage 2"


def test_two_intervals_opening_before_the_sample_are_refused_not_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "membership.parquet"
    _membership_frame([("AAA", "1990-01-01", "2026-06-16"), ("AAA", "1995-01-01", "2025-01-01")]).to_parquet(
        path, index=False
    )
    with pytest.raises(LegacyImportError, match="defect in the source"):
        _membership_rows(
            path, ["AAA"], index_id="sp500", first_session=dt.date(2020, 1, 2), last_session=dt.date(2026, 6, 15)
        )


def test_the_import_keeps_only_exchange_sessions() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-02", tz="America/New_York"),  # session
            pd.Timestamp("2026-01-01", tz="America/New_York"),  # New Year's Day
            pd.Timestamp("2026-01-05", tz="America/New_York"),  # session
        ]
    )
    frames = {"AAA": pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=index)}
    kept, dropped = session_index(frames)
    assert [ts.date().isoformat() for ts in kept] == ["2026-01-02", "2026-01-05"]
    assert dropped == 1


# ---------------------------------------------------------------------------
# The golden reproduction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DATASET / "manifest.json").is_file(),
    reason=(
        "the reproduction dataset is absent: the vendor bar caches are not redistributable and are not in the "
        "repository. Build it with `python -m prism.scripts.reproduce --case demotion_b1 --recorded "
        "<repo>/results/demotion_b1 --cache-dir <repo>/data`."
    ),
)
def test_the_first_fold_reproduces_the_recorded_fold_within_the_declared_tolerance(tmp_path: Path) -> None:
    """One fold of the recorded evaluation, end to end through the evaluator.

    One fold rather than twenty-one because the property under test is the
    engine's agreement with the recorded driver, which a single fold exercises
    completely — the fit boundary, the cadence, the closed-form band from the
    formation replay, the bucket spread, the reset flatten and the cost
    accounting — while twenty-one would cost ten minutes of suite time to
    re-establish the same fact. The whole-record comparison is the recorded
    reproduction under ``results/demotion_b1_reproduction``.
    """
    config = _recorded_config()
    recorded = _recorded_folds()[0]["metrics"]
    workspace = Workspace(root=tmp_path).ensure()
    for name in ("examples", "folds"):
        (workspace.root / name).symlink_to(REPO / name)

    table = FoldTable.from_toml(REPO / FOLD_TABLE)
    one = FoldTable(table_id="demotion_b1_fold0", folds=table.folds[:1], source=table.source, timezone=table.timezone)
    spec = load_spec(workspace.root / ARMS["bucket"], workspace=workspace)
    result = walk_forward_evaluate(
        spec,
        workspace=workspace,
        dataset=Dataset.open(DATASET),
        fold_table=one,
        kind="reproduction",
        selection_set_id="golden",
        run_channel="engineering",
        execution=ExecutionConfig(**config["execution"]),
        fold_portfolio="reset",
        cost_refresh="once",
        capital=float(config["initial_capital"]),
    )
    produced = dict(result.metrics)
    for key in ("total_return", "sharpe", "total_cost", "avg_turnover", "avg_gross", "annualized_vol", "psr"):
        assert produced[key] == pytest.approx(recorded[key], rel=FOLD_TOLERANCE, abs=FOLD_TOLERANCE), key
    assert len(result.scored.returns) == recorded_test_rows()


def recorded_test_rows() -> int:
    return int(_recorded_folds()[0]["test_rows"])


# ---------------------------------------------------------------------------
# The stage 3a exit: the legacy import round-trips the momentum panel exactly
# ---------------------------------------------------------------------------


#: Where the range-encoded vendor caches live. They are not redistributable and
#: are not in the repository (§3.2, "Non-redistributable inputs"), so the path
#: is resolved rather than assumed: this checkout's ``data`` directory first,
#: then ``PRISM_BAR_CACHE_DIR`` for an operator whose caches sit elsewhere --
#: which is the case in a worktree, where ``data`` holds only the built frozen
#: dataset. Hard-coding one machine's path would have made the test skip
#: everywhere else while reading green.
def _bar_cache_dir() -> Path | None:
    candidates = [REPO / "data"]
    declared = os.environ.get("PRISM_BAR_CACHE_DIR")
    if declared:
        candidates.insert(0, Path(declared))
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*_1d_*.parquet")):
            return candidate
    return None


@pytest.mark.skipif(
    _bar_cache_dir() is None,
    reason=(
        "the range-encoded vendor bar caches are absent: they are not redistributable and are not in the "
        "repository. Point PRISM_BAR_CACHE_DIR at a directory holding data/{SYMBOL}_1d_{start}_{end}.parquet."
    ),
)
def test_the_legacy_import_round_trips_the_recorded_momentum_panel_exactly(tmp_path: Path) -> None:
    """Stage 3a's fifth exit condition, on the real 574-name panel rather than a fixture.

    ``tests/test_import_legacy.py`` proves the round-trip property on built
    caches whose disagreement is planted; it deliberately reads no real cache
    directory. That leaves the exit condition itself -- *the* momentum panel,
    the one the certified B1 record consumed -- pinned by nothing, so a change
    to the import could shift a real cell and every fixture would still pass.
    This is that missing pin.

    The window ends on the last session the recorded sample covers rather than
    on the configuration's ``end_date``. The two differ: fifteen symbols carry a
    2026-06-16 bar in ``data/store`` that the loader's selected range cache does
    not, so over the configuration's own window the import holds fifteen rows
    the loader lacks. Those are extra data and not a disagreement, but the exit
    condition is a statement about the panel the record was computed on, and
    that panel stops at 2026-06-15.
    """
    cache_dir = _bar_cache_dir()
    assert cache_dir is not None
    config = _recorded_config()
    symbols = list(config["symbols"])
    assert len(symbols) == 574, "the recorded universe is the 574-name cross-section"

    root = tmp_path / "imported"
    import_legacy(root, cache_dir=cache_dir, vintage="round_trip", symbols=symbols)

    mapping = sid_by_symbol(root)
    assert not [s for s in symbols if s not in mapping], "every recorded symbol must have been assigned a sid"
    assert all(sid.startswith("sec_") for sid in mapping.values()), "a ticker is never a sid (decision D2)"

    for field in ("close", "open", "high", "low", "volume"):
        trip = round_trip_panel(
            root,
            cache_dir=cache_dir,
            symbols=symbols,
            start=str(config["start_date"]),
            end=RECORDED_LAST_SESSION,
            field=field,
        )
        assert trip.max_abs_difference == 0.0, (field, trip.disagreements[:5])
        assert not trip.disagreements, (field, trip.disagreements[:5])
        assert not trip.only_loader, (field, trip.only_loader[:5])
        assert not trip.only_imported, (field, trip.only_imported[:5])
        assert not trip.missing_symbols, (field, trip.missing_symbols[:5])
        assert trip.exact, (field, trip.to_dict())
        assert trip.n_cells == RECORDED_PANEL_CELLS, (field, trip.n_cells)


def test_the_recorded_headline_this_reproduction_targets_is_the_summary_the_record_carries() -> None:
    """``RECORDED_HEADLINE`` is read off the record, not typed in from prose."""
    summary = json.loads((RECORDED / "summary.json").read_text())
    for key, value in RECORDED_HEADLINE.items():
        assert summary[key] == value, key
