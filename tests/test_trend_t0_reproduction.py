"""The stage 3b reproduction of the recorded T0 trend record (spec section 5).

Section 5 schedules this case to stage 3 by name, because it needs capabilities
the data layer does not deliver until then: a ``total_return`` scoring basis
with cash distributions applied to the scoring series while costs stay on the
as-traded close, a per-fold bucket spread frame keyed to the frozen fold table,
a fixed registered list universe, inverse-volatility sizing, and a session
table reaching 2006. All of them are exercised here.

Four kinds of check live in this file, and only the last two need the untracked
bar caches.

*The specification is the recorded configuration.* ``examples/trend/trend_t0.toml``
is asserted field by field against ``results/trend_t0/config.json``, which is
tracked. A value that drifts on either side fails here rather than quietly
reproducing a different configuration under the recorded name.

*The geometry is the recorded geometry.* ``folds/trend_t0.toml`` is asserted to
hash to the content hash the recorded ``folds.json`` produces, and to carry the
same 74 windows date for date.

*The mechanics this case added.* Resolving a literal symbol list onto a dataset
that mints identifiers, pinning an import to the cache files a record names,
and evaluating a span the caller declared over a manifest that records no
coverage — each checked against a hand-built case whose expected values are
written out independently of the code under test.

*The golden reproduction.* The evaluator's own per-fold metrics for the first
three folds are compared with the metrics ``results/trend_t0/folds.json``
records, at the tolerance declared in ``FOLD_TOLERANCE`` below. Three folds
rather than one because the per-fold bucket cadence is the capability under
test and a single fold cannot show it: fold 0 prices two names at 1 bp and fold
2 prices three, so a run that silently froze the first fold's vector for the
whole sample -- which is what the momentum sibling does and what this case
deliberately does not -- would fail here. It skips when the reproduction
dataset is absent, because the vendor bar caches are not redistributable and
are not in the repository; the command that builds it is in the skip message.

The recorded cell's own conclusion is negative -- the sleeve carries no
standalone edge and its convexity admission was refused -- so a faithful
reproduction reproduces a negative result. That is what these assertions pin,
and nothing here may be read as evidence of an edge.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.evaluate import EvaluationError, _listed_universe, _span, _Visibility, walk_forward_evaluate
from prism.io.dataset import Dataset
from prism.io.legacy_import import LegacyImportError, discover_legacy_inputs
from prism.scripts.reproduce import CASES
from prism.spec import load_spec
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace

REPO = Path(__file__).resolve().parents[1]

#: The recorded record, read from THIS checkout. Its five JSON artifacts
#: (``config.json``, ``folds.json``, ``summary.json``, ``claim_packet.json``,
#: ``adjudication.json``) are tracked, so every assertion below that reads them
#: runs wherever the repository is cloned. The four CSV time series in the same
#: directory are untracked by design and nothing here reads them.
RECORDED = REPO / "results" / "trend_t0"
DATASET = REPO / "data" / "frozen" / "trend_t0"

CASE = CASES["trend_t0"]

#: The declared tolerance for the golden reproduction, and the reason for the
#: number. The evaluator and the recorded driver sum the same per-bar cost and
#: return terms over the same ten-name cross-section in a different association
#: order, so the two differ by floating-point association and by nothing else
#: that is intended. The one input that is not bit-identical is the
#: total-return scoring panel, a cumulative product whose recorded and rebuilt
#: forms agree to 3.2e-16 relative (Run, over all 4,981 x 10 cells); it feeds a
#: ratio score whose *sign* is what construction consumes, so it can only move
#: a weight for a name whose score sits within 1e-15 of zero. 1e-9 relative is
#: six orders of magnitude of headroom over that and still four orders below
#: the smallest economically meaningful difference on these figures. The
#: absolute floor is tighter than the relative one deliberately: the recorded
#: annualized return is 7.6e-4 and the recorded periodic Sharpe 2.7e-3, and a
#: pure relative test on a near-zero figure asks for more precision than the
#: arithmetic can supply. A difference above this is an engine difference and
#: is reported as a finding, never absorbed.
FOLD_TOLERANCE = 1e-9
ABS_TOLERANCE = 1e-12

#: How many folds the golden slice runs, and why that number: fold 0 and fold 2
#: carry different spread-bucket vectors, so the slice exercises the per-fold
#: cadence rather than merely one fold's arithmetic.
GOLDEN_FOLDS = 3

#: Run figures from the recorded record, pinned so a silent change -- a
#: truncated import, a dropped symbol, a session table that moved -- fails
#: rather than reading green.
RECORDED_SESSIONS = 4981
RECORDED_TEST_ROWS = 4662
RECORDED_FIRST_SESSION = "2006-09-27"
RECORDED_LAST_SESSION = "2026-07-17"


def _recorded_config() -> dict:
    # Not a skip: config.json is tracked, so its absence is a damaged checkout
    # and not an environment this suite may quietly pass in.
    path = RECORDED / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"the tracked recorded record {path} is missing from this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def _recorded_folds() -> list[dict]:
    return json.loads((RECORDED / "folds.json").read_text(encoding="utf-8"))


def _recorded_summary() -> dict:
    return json.loads((RECORDED / "summary.json").read_text(encoding="utf-8"))


def _spec():
    workspace = Workspace(root=REPO)
    return load_spec(REPO / CASE.arms["bucket"], workspace=workspace)


# ---------------------------------------------------------------------------
# The specification is the recorded configuration
# ---------------------------------------------------------------------------


def test_the_specification_declares_the_recorded_universe() -> None:
    config = _recorded_config()
    spec = _spec()
    assert spec.strategy.universe_index is None
    assert sorted(spec.strategy.universe_list or ()) == sorted(config["symbols"])
    assert len(spec.strategy.universe_list or ()) == len(config["symbols"]) == 10
    assert spec.strategy.universe_as_of == config["universe"]["asof"]
    assert config["universe"]["policy"] == "fixed_registered_list"
    assert spec.strategy.renormalize is False
    assert spec.strategy.combination == "position_space"


def test_the_specification_declares_the_recorded_signal_and_sizing() -> None:
    config = _recorded_config()
    spec = _spec()
    (component,) = spec.components
    assert component.spec.path == "prism.signal.trend_node:TrendSignalNode"
    assert component.spec.params["lookback_bars"] == config["signal"]["lookback_bars"]
    assert component.spec.params["skip_bars"] == config["signal"]["skip_bars"]
    assert component.spec.update_every_bars == config["signal"]["decision_every"]
    assert component.spec.valid_for_bars == config["signal"]["decision_every"]
    assert component.spec.horizon_bars == config["signal"]["decision_every"]
    # `construction.sizing = "inverse_vol"` is the `sign_inverse_vol` rule.
    assert config["construction"]["sizing"] == "inverse_vol"
    assert component.spec.to_weights == "sign_inverse_vol"
    weights = component.spec.to_weights_params
    assert weights["vol_ewma_bars"] == config["construction"]["vol_ewma_bars"]
    assert weights["annualization_bars"] == config["construction"]["annualization_bars"]
    assert weights["max_gross"] == config["walk"]["max_gross"]
    assert weights["max_symbol_abs_weight"] == config["walk"]["max_symbol_abs_weight"]
    # The basis is the node's own declaration, never taken from the file, and
    # it is the recorded `price_convention`.
    assert component.declarations["scoring_basis"] == "total_return"
    assert config["price_convention"] == "total_return"
    assert config["data_convention"] == "split_adjusted_total_return_dividends_as_cash"


def test_the_specification_declares_the_recorded_construction() -> None:
    config = _recorded_config()
    spec = _spec()
    options = dict(spec.construction.options)
    assert spec.construction.kind == "heuristic"
    assert options["step_order"] == "construct_band_cap_participation"
    assert options["max_gross"] == config["walk"]["max_gross"]
    assert options["max_symbol_abs_weight"] == config["walk"]["max_symbol_abs_weight"]
    assert options["band"]["mode"] == config["walk"]["band_mode"] == "closed_form"
    assert options["band"]["gamma_risk"] == config["band_gamma_risk"]
    assert options["participation"]["max_participation"] == config["walk"]["max_participation"]
    assert options["participation"]["window_bars"] == config["signal"]["dollar_volume_window"]
    assert options["participation"]["statistic"] == "median"
    # NOT `execution.adv_floor_dollars`: the recorded gate passes adv_floor=0.0
    # explicitly, because flooring ADV up loosens the bound for exactly the
    # illiquid names it protects.
    assert options["participation"]["floor_dollars"] == 0.0
    assert config["execution"]["adv_floor_dollars"] == 100000.0
    assert options["participation"]["base"] == "initial_capital"
    assert options["spread"]["mode"] == config["walk"]["spread_mode"] == "bucket"


def test_the_listing_rule_is_the_recorded_entry_rule() -> None:
    """`entry_bars = 252` is a universe rule and is declared as listing seasoning."""
    spec = _spec()
    assert spec.strategy.eligibility.listing_seasoning_bars == 252
    # Every other gate is off: the recorded driver screens this fixed list on
    # nothing else, and a name without lookback history simply scores NaN.
    assert spec.strategy.eligibility.min_price == 0.0
    assert spec.strategy.eligibility.min_median_dollar_volume == 0.0
    assert spec.strategy.eligibility.return_history_bars == 0
    assert spec.strategy.eligibility.membership_required is False
    assert spec.strategy.eligibility.exclude_asset_classes == ()


def test_the_registered_case_charges_the_recorded_cost_cadence() -> None:
    config = _recorded_config()
    assert config["spread_accounting"] == "per_fold_bucket"
    # `per_fold_bucket` is `cost_refresh = "fold"`, and it is NOT the momentum
    # sibling's cadence. Freezing the first fold's vector was inert on a 2020+
    # large-cap panel and is not inert here.
    assert CASE.cost_refresh == "fold"
    assert CASES["demotion_b1"].cost_refresh == "once"
    assert config["execution"] == {
        field: getattr(ExecutionConfig(**config["execution"]), field)
        for field in ExecutionConfig.__dataclass_fields__  # type: ignore[attr-defined]
    }


# ---------------------------------------------------------------------------
# The geometry is the recorded geometry
# ---------------------------------------------------------------------------


def test_the_fold_table_is_the_recorded_geometry() -> None:
    table = FoldTable.from_toml(REPO / CASE.fold_table)
    recorded = _recorded_folds()
    assert table.content_hash == CASE.fold_table_sha256
    assert len(table) == len(recorded) == 74
    for boundary, row in zip(table.folds, recorded):
        assert boundary.index == row["fold"]
        for field, key in (
            ("formation_start", "formation_start"),
            ("formation_end", "formation_end"),
            ("test_start", "test_start"),
            ("test_end", "test_end"),
        ):
            assert str(getattr(boundary, field).date()) == row[key][:10], (row["fold"], key)


def test_the_recorded_sample_is_the_one_the_summary_counts() -> None:
    config = _recorded_config()
    summary = _recorded_summary()
    assert str(config["start_date"])[:10] == RECORDED_FIRST_SESSION
    assert str(config["end_date"])[:10] == RECORDED_LAST_SESSION
    assert int(summary["n_sessions"]) == RECORDED_SESSIONS
    assert int(summary["n_test_rows"]) == RECORDED_TEST_ROWS
    # The bundled session table has to cover the recorded sample exactly, which
    # is why decision D12 widened it to 2006. If it does not, the reproduction
    # is scoring a different calendar and that is a finding against the table.
    from prism.io import sessions

    grid = sessions.sessions_between(
        dt.date.fromisoformat(RECORDED_FIRST_SESSION), dt.date.fromisoformat(RECORDED_LAST_SESSION)
    )
    assert len(grid) == RECORDED_SESSIONS


def test_the_headline_this_runner_compares_against_is_the_recorded_summary() -> None:
    """Every pinned headline figure is the recorded one, read from the tracked summary."""
    summary = _recorded_summary()
    assert CASE.headline
    for key, value in CASE.headline.items():
        assert key in summary, key
        assert float(summary[key]) == pytest.approx(value, rel=0.0, abs=0.0), key
    # The recorded conclusion is negative and stays negative: an annualized
    # Sharpe under its own hurdle. Pinning it here means a reproduction that
    # drifted upward would fail rather than read as an improvement.
    assert float(summary["sharpe"]) < float(summary["after_cost_hurdle"]["periodic_sharpe_hurdle"]) * 252**0.5
    assert float(summary["oos_periodic_sharpe"]) < float(summary["after_cost_hurdle"]["periodic_sharpe_hurdle"])


# ---------------------------------------------------------------------------
# The mechanics this case added
# ---------------------------------------------------------------------------


def test_an_explicitly_named_span_needs_no_manifest_coverage() -> None:
    """A caller that named both bounds has answered the question the manifest would."""

    class _Manifest:
        id = "test"
        coverage: dict = {}

    class _Dataset:
        manifest = _Manifest()

    assert _span(_Dataset(), "2006-09-27", "2026-07-17") == ("2006-09-27", "2026-07-17")
    with pytest.raises(EvaluationError, match="records no coverage.sessions"):
        _span(_Dataset(), None, "2026-07-17")


def test_a_pinned_import_reads_the_named_files_and_only_those(tmp_path: Path) -> None:
    """The record names twenty cache files; the glob would sweep a dozen more."""
    for name in (
        "SPY_1d_min_2026-08-14.parquet",
        "SPY_1d_2018-01-01_2026-07-22.parquet",
        "SPY_1d_min_2026-08-15.parquet",  # a later file the recorded run never read
        "EEM_1d_2007-01-01_2017-12-31.parquet",
    ):
        (tmp_path / name).write_bytes(b"")
    pinned = discover_legacy_inputs(
        tmp_path,
        cache_files={
            "SPY": ["SPY_1d_min_2026-08-14.parquet", "SPY_1d_2018-01-01_2026-07-22.parquet"],
            "EEM": ["EEM_1d_2007-01-01_2017-12-31.parquet"],
        },
    )
    assert [p.name for p in pinned.files["SPY"]] == [
        "SPY_1d_min_2026-08-14.parquet",
        "SPY_1d_2018-01-01_2026-07-22.parquet",
    ]
    assert [p.name for p in pinned.files["EEM"]] == ["EEM_1d_2007-01-01_2017-12-31.parquet"]
    assert pinned.pinned_files == (
        "EEM:EEM_1d_2007-01-01_2017-12-31.parquet",
        "SPY:SPY_1d_min_2026-08-14.parquet",
        "SPY:SPY_1d_2018-01-01_2026-07-22.parquet",
    )
    # Discovery, by contrast, sweeps the later file in -- which is exactly the
    # silent panel change the pin exists to prevent.
    discovered = discover_legacy_inputs(tmp_path)
    assert len(discovered.files["SPY"]) == 3
    assert discovered.pinned_files == ()


def test_a_pinned_import_refuses_a_missing_or_misfiled_name(tmp_path: Path) -> None:
    (tmp_path / "SPY_1d_min_2026-08-14.parquet").write_bytes(b"")
    with pytest.raises(LegacyImportError, match="not a file under"):
        discover_legacy_inputs(tmp_path, cache_files={"SPY": ["SPY_1d_min_2026-08-13.parquet"]})
    with pytest.raises(LegacyImportError, match="names 'SPY'"):
        discover_legacy_inputs(tmp_path, cache_files={"EEM": ["SPY_1d_min_2026-08-14.parquet"]})
    with pytest.raises(LegacyImportError, match="empty cache-file list"):
        discover_legacy_inputs(tmp_path, cache_files={"SPY": []})


# ---------------------------------------------------------------------------
# The golden reproduction
# ---------------------------------------------------------------------------


SKIP_REASON = (
    "the reproduction dataset is absent: the vendor bar caches are not redistributable and are not in the "
    "repository. Build it with `python -m prism.scripts.reproduce --case trend_t0 --recorded "
    "<repo>/results/trend_t0 --cache-dir <bar cache dir>`."
)
needs_dataset = pytest.mark.skipif(not (DATASET / "manifest.json").is_file(), reason=SKIP_REASON)


@needs_dataset
def test_a_literal_symbol_list_resolves_onto_the_datasets_own_sids() -> None:
    """The declaration is a symbol list; this dataset mints ``sec_%06d`` identifiers."""
    spec = _spec()
    dataset = Dataset.open(DATASET)
    visibility = _Visibility(mode="research_vintage", vintage=dataset.manifest.vintage)
    grid = [dt.date.fromisoformat(RECORDED_FIRST_SESSION), dt.date.fromisoformat(RECORDED_LAST_SESSION)]
    resolved = _listed_universe(spec, dataset, visibility, grid, list(spec.strategy.universe_list or ()))
    assert len(resolved) == 10
    assert all(sid.startswith("sec_") for sid in resolved)
    view = visibility.view(dataset, RECORDED_LAST_SESSION)
    listings = view.listings(resolved).set_index("sid")["ticker"].astype(str)
    assert [listings.loc[sid] for sid in resolved] == list(spec.strategy.universe_list or ())
    # A sid the manifest already carries bars for passes through untouched, so
    # a ticker-keyed dataset (the momentum reproduction's) is unaffected.
    assert _listed_universe(spec, dataset, visibility, grid, [resolved[0]]) == [resolved[0]]
    # A name the dataset cannot answer for is a refusal, never a smaller book.
    with pytest.raises(EvaluationError, match="neither as a sid nor as a ticker"):
        _listed_universe(spec, dataset, visibility, grid, ["NOT_A_TICKER"])
    # Two declarations naming one instrument would double its weight.
    with pytest.raises(EvaluationError, match="resolves .* more than once"):
        _listed_universe(spec, dataset, visibility, grid, ["SPY", resolved[0]])


@needs_dataset
def test_the_recorded_entry_sessions_reproduce_from_the_listing_rule() -> None:
    """`listing_seasoning_bars = 252` lands on the ten recorded entry dates."""
    from prism.evaluate import _seasoned

    dataset = Dataset.open(DATASET)
    visibility = _Visibility(mode="research_vintage", vintage=dataset.manifest.vintage)
    recorded = {k: v[:10] for k, v in _recorded_summary()["entry_sessions"].items()}
    grid = [dt.date.fromisoformat(RECORDED_FIRST_SESSION), dt.date.fromisoformat(RECORDED_LAST_SESSION)]
    spec = _spec()
    sids = _listed_universe(spec, dataset, visibility, grid, list(spec.strategy.universe_list or ()))
    tickers = list(spec.strategy.universe_list or ())
    for sid, ticker in zip(sids, tickers):
        entry = recorded[ticker]
        view = visibility.view(dataset, entry)
        assert bool(_seasoned(view, [sid], 252).loc[sid]), f"{ticker} is not seasoned on its recorded entry {entry}"
        before = view.panel([sid], entry, "close").index[-2].date().isoformat()
        assert not bool(_seasoned(visibility.view(dataset, before), [sid], 252).loc[sid]), ticker


@needs_dataset
def test_the_first_folds_reproduce_the_recorded_folds_within_the_declared_tolerance(tmp_path: Path) -> None:
    """Three folds of the recorded evaluation, end to end through the evaluator.

    Three rather than seventy-four because the property under test is the
    engine's agreement with the recorded driver, which three folds exercise
    completely -- the total-return scoring basis, the listing rule, the
    cadence, the closed-form band from the formation replay, the *per-fold*
    bucket spread, the reset flatten and the cost accounting -- while
    seventy-four would cost two minutes of suite time to re-establish the same
    fact. The whole-record comparison is the recorded reproduction run.
    """
    config = _recorded_config()
    recorded = _recorded_folds()[:GOLDEN_FOLDS]
    workspace = Workspace(root=tmp_path).ensure()
    for name in ("examples", "folds"):
        (workspace.root / name).symlink_to(REPO / name)

    table = FoldTable.from_toml(REPO / CASE.fold_table)
    sliced = FoldTable(
        table_id="trend_t0_golden",
        folds=table.folds[:GOLDEN_FOLDS],
        source=table.source,
        timezone=table.timezone,
    )
    spec = load_spec(workspace.root / CASE.arms["bucket"], workspace=workspace)
    result = walk_forward_evaluate(
        spec,
        workspace=workspace,
        dataset=Dataset.open(DATASET),
        fold_table=sliced,
        kind="reproduction",
        selection_set_id="golden",
        run_channel="engineering",
        execution=ExecutionConfig(**config["execution"]),
        fold_portfolio="reset",
        cost_refresh=CASE.cost_refresh,
        capital=float(config["initial_capital"]),
        start=str(config["start_date"])[:10],
        end=str(config["end_date"])[:10],
        reconcile=False,
    )
    assert len(result.scored.returns) == sum(int(row["test_rows"]) for row in recorded)

    produced = {int(fold["fold"]): fold for fold in result.record.evaluation["folds"]}
    for row in recorded:
        mine = produced[int(row["fold"])]
        assert mine["train"]["start"] == row["formation_start"][:10]
        assert mine["train"]["end"] == row["formation_end"][:10]
        assert mine["test"]["start"] == row["test_start"][:10]
        assert mine["test"]["end"] == row["test_end"][:10]
        for recorded_key, our_key in (
            ("total_return", "total_return"),
            ("total_cost", "total_cost"),
            ("avg_gross", "avg_gross"),
            ("avg_net", "avg_net"),
            ("avg_turnover", "avg_turnover"),
            ("sharpe", "annualized_sharpe"),
        ):
            assert mine["metrics"][our_key] == pytest.approx(
                row["metrics"][recorded_key], rel=FOLD_TOLERANCE, abs=ABS_TOLERANCE
            ), (row["fold"], recorded_key)

    # The per-fold cost inputs are the recorded ones, and they move between
    # folds: freezing fold 0's vector would put six names at 10 bps for the
    # whole sample, which is the momentum sibling's cadence and not this one's.
    per_fold = {int(entry["fold"]): entry for entry in result.record.costs["cost_refresh"]["per_fold"]}
    assert result.record.costs["cost_refresh"]["cadence"] == "fold"
    for row in recorded:
        entry = per_fold[int(row["fold"])]
        counts: dict[float, int] = {}
        for value in row["spread_bps"].values():
            counts[float(value)] = counts.get(float(value), 0) + 1
        assert {float(k): int(v) for k, v in entry["spread_bps"]["buckets"].items()} == counts, row["fold"]
        band = pd.Series(row["band"], dtype=float)
        positive = band[band > 0.0]
        assert entry["band"]["n_positive"] == len(positive), row["fold"]
        assert entry["band"]["max"] == pytest.approx(float(band.max()), rel=FOLD_TOLERANCE, abs=ABS_TOLERANCE)
        if len(positive):
            assert entry["band"]["median"] == pytest.approx(
                float(positive.median()), rel=FOLD_TOLERANCE, abs=ABS_TOLERANCE
            )
    assert per_fold[0]["spread_bps"]["buckets"] != per_fold[GOLDEN_FOLDS - 1]["spread_bps"]["buckets"]
