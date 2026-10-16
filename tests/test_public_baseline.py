"""Synthetic-data tests for the public baseline driver (``research/scripts/public_baseline.py``).

Nothing here reads market data. The registration the driver will run is checked in
the shape ``docs/public_baseline_design.md`` section 2b gives it, with placeholder
hashes; the end-to-end test runs a scaled-down registration of the same shape through
the canonical engine on a synthetic dataset and fake French files.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prism.io import sessions
from prism.io.dataset import Dataset
from prism.io.synthetic import EQUITY_INDEX, write_synthetic_dataset
from prism.spec import load_spec
from prism.validation.capacity import CrowdScenario, crowd_impact_cost
from prism.validation.hurdle import tbill_hurdle
from prism.validation.metrics import expected_max_sharpe_with_n
from prism.validation.trials import TrialLog
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace
from research.scripts import public_baseline as pb
from research.scripts import universe_partition

LANE = Path(__file__).resolve().parents[1]
MANIFEST_ID = "bedf8c4c0f0c2195ce5aace938f4ad38e23e6b69b57075080073004692d46eef"

# --------------------------------------------------------------------------- the planned registration

PLANNED = """
id = "2026-10-public-baseline"
created = 2026-10-16
selection_set = "public_baseline_v1"
route = "replication"
trial_budget = 30
candidate = "T21"

[source]
citation_characteristic = "Jegadeesh and Titman (1993)"
citation_sizing = "Barroso and Santa-Clara (2015)"
publication = "2015-04"
target_vol = 0.12
lookback_sessions = 126
refresh_sessions = 21
max_scale = 2.0

[reference]
momentum_source = "ff_mom_daily"
momentum_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
momentum_sha256 = "{zeros}"
momentum_bytes = 1
momentum_last_modified = "placeholder"
momentum_vintage_line = "placeholder"
cash_source = "ff3_daily"
cash_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
cash_sha256 = "{zeros}"
cache_dir = "data/public/vintages"
warmup_sessions = 1260

[criteria]
publication_bar_t = 3.0
publication_bar_min_years = 20.0
post_publication_min_years = 10.0
pre_publication_haircut = 0.4
fidelity_floor = 0.75
hurdle_window_sessions = 252
sessions_per_year = 252
vol_basis_rule = "larger_hurdle_of_target_and_realized"
sleeve_buffer = 0.02
sleeve_fee_annual = 0.0010
buffer_cash_yield = 0.0
reported_cash_yields = [0.0, 0.033]
parity_min_daily_corr = 0.99
parity_max_abs_sharpe_diff = 0.05
tranching_paired_lower_bound = 0.0

[implementation]
dataset = {{ path = "data/frozen/demotion_b1", manifest_id = "{manifest}", vintage = "demotion_b1_2026-06-16" }}
signal = {{ node = "prism.signal.momentum_node.MomentumSignalNode", lookback = 252, skip = 21 }}
construction = {{ rule = "decile_neutral", decile = 0.10, max_symbol_abs_weight = 0.10, raw_gross = 1.0 }}
folds = {{ test_sessions = 63, formation_sessions = 312 }}
grid_epoch = "2006-01-03"
p00_anchor = "2022-03-29"
initial_capital = 100000.0
share_increment = 1.0
fill_model = "full_next_open"
cost_stack = "examples/reference/costs/retail_v3.toml"
scored_last = "2026-06-12"
parity_series = "results/2026-09-vol-momentum/series.csv"

[guards]
min_valid_returns = 126
max_zero_return_fraction = 0.05
require_decision_bar_is_last_bar = true

[rows]
phases = {{ ids = "P00..P20", tranches = 1, offsets = "0..20" }}
T21 = {{ tranches = 21 }}
D-split = {{ tranches = 21, fill_model = "split_open_close" }}
D-liq = {{ tranches = 21, leg_weights = "sqrt_median_dollar_volume_20" }}
D-part = {{ tranches = 21, keep_fraction = 0.8, seeds = [11, 22, 33] }}
T21-25k = {{ tranches = 21, initial_capital = 25000.0 }}
T21-1m = {{ tranches = 21, initial_capital = 1000000.0 }}
T3-25k = {{ tranches = 3, offsets = "r, r+7, r+14", initial_capital = 25000.0 }}

[bootstrap]
block_bars = 42
n_resamples = 2000
seed = 20261015
confidence = 0.95

[kill]
first_read_min_sessions = 126
read_every_sessions = 63
max_drawdown = 0.25
min_forward_fidelity = 0.5
economics_consecutive_reads = 2
monitor = {{ method = "empirical_bernstein", bound = 0.1, alpha = 0.05, report_only = true }}

[capacity]
follower_capital = [0.0, 1e6, 1e7, 1e8, 1e9, 1e10]
window_share = {{ open_low = 0.01, open_high = 0.03, close = 0.08, full_day = 1.0 }}
synchronization = ["P00:1", "P00:1/21", "T21:1"]
impact_y = [0.5, 0.7, 1.0]
sigma = "trailing_20_session_daily"
selection_forbidden = true
"""


def planned(**replace: str) -> str:
    text = PLANNED.format(zeros="0" * 64, manifest=MANIFEST_ID)
    for old, new in replace.items():
        assert old in text, old
        text = text.replace(old, new)
    return text


def parse(text: str) -> pb.Baseline:
    return pb.parse_registration(tomllib.loads(text), path=Path("registration.toml"), sha256="0" * 64)


def test_the_planned_registration_resolves_to_thirty_rows_on_the_grid() -> None:
    baseline = parse(planned())
    ids = [row.id for row in baseline.rows]
    assert len(ids) == 30 and len(set(ids)) == 30
    base = sessions.grid_position(dt.date(2022, 3, 29)) % 21
    assert baseline.base_offset == base
    phases = [row for row in baseline.rows if row.group == "phases"]
    assert [row.id for row in phases] == [f"P{k:02d}" for k in range(21)]
    assert [row.offset for row in phases] == [(base + k) % 21 for k in range(21)]
    assert all(row.tranches == 1 for row in phases)
    assert [row.id for row in baseline.run_order()[:3]] == ["P00", "T21", "P01"]
    assert [row.id for row in baseline.rows if row.partitioned] == ["D-part1", "D-part2", "D-part3"]
    liquid = baseline.row("D-liq")
    assert liquid.to_weights == "decile_neutral_sqrt_dv" and liquid.dollar_volume_window == 20
    assert baseline.row("D-split").fill_model == "split_open_close"
    assert baseline.row("T3-25k").tranches == 3 and baseline.row("T3-25k").capital == 25_000.0
    assert [row.id for row in pb.account_size_rows(baseline)] == ["T21-25k", "T21-1m", "T3-25k"]
    assert [row.id for row in pb.diagnostic_rows(baseline)] == ["D-split", "D-liq", "D-part1", "D-part2", "D-part3"]


@pytest.mark.parametrize(
    "replace, message",
    [
        ({"trial_budget = 30": "trial_budget = 29"}, "trial_budget"),
        ({"offsets = \"r, r+7, r+14\"": "offsets = \"r, r+6, r+14\""}, "tranche_plans"),
        ({"fidelity_floor = 0.75": "fidelity_floor = 0.75\nfidelity_flor = 0.8"}, "unknown key"),
        ({"min_valid_returns = 126": "min_valid_returns = 100"}, "min_valid_returns"),
        ({"grid_epoch = \"2006-01-03\"": "grid_epoch = \"2006-01-04\""}, "GRID_EPOCH"),
        ({"p00_anchor = \"2022-03-29\"": "p00_anchor = \"2022-03-27\""}, "sessions"),
        ({"candidate = \"T21\"": "candidate = \"T22\""}, "candidate"),
        ({"selection_forbidden = true": "selection_forbidden = false"}, "selection_forbidden"),
        ({"\"P00:1/21\"": "\"P99:1\""}, "synchronization"),
        ({"folds = { test_sessions = 63, formation_sessions = 312 }\n": ""}, "folds"),
    ],
)
def test_the_loader_refuses_what_it_cannot_run(replace, message) -> None:
    with pytest.raises(pb.BaselineError, match=message):
        parse(planned(**replace))


def test_the_registered_fold_geometry_reproduces_the_recorded_folds(tmp_path) -> None:
    """{63, 312} from 2022-03-29 is folds 4-20 of folds/demotion_b1.toml, which vol_momentum_v1 was charged under."""
    built = pb.build_fold_table(parse(planned()))
    recorded = FoldTable.from_toml(LANE / "folds" / "demotion_b1.toml")
    tail = [fold for fold in recorded.folds if fold.test_start >= pd.Timestamp("2022-03-29", tz="America/New_York")]
    assert len(built.folds) == len(tail) == 17
    for ours, theirs in zip(built.folds, tail):
        assert (ours.formation_start, ours.formation_end, ours.test_start, ours.test_end) == (
            theirs.formation_start, theirs.formation_end, theirs.test_start, theirs.test_end)
    single = pb.build_fold_table(parse(planned(**{"test_sessions = 63": "test_sessions = 0"})))
    assert single.n_folds == 1 and single.folds[0].test_end == built.folds[-1].test_end
    assert FoldTable.from_toml(_write(tmp_path / "folds.toml", built.to_toml())).content_hash == built.content_hash


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_every_row_specification_loads_and_has_its_own_identity(tmp_path) -> None:
    baseline = parse(planned())
    workspace = Workspace(LANE)
    seen = {}
    for row_id in ("P00", "P07", "T21", "D-split", "D-liq", "T3-25k"):
        declared = pb.row_specification(baseline, baseline.row(row_id))
        text = pb.render_spec_toml(declared["specification"])
        assert tomllib.loads(text) == json.loads(json.dumps(declared["specification"]))
        path = _write(tmp_path / f"spec-{row_id}.toml", text)
        spec = load_spec(path, workspace=workspace)
        assert spec.sizing is not None and spec.sizing.max_gross == 2.0 and spec.sizing.lookback_bars == 126
        assert spec.construction.options["max_gross"] == 2.0 and spec.construction.options["band"] == {"mode": "off"}
        component = spec.components[0].spec
        assert component.grid_schedule == (baseline.row(row_id).tranches, baseline.row(row_id).offset)
        assert spec.policy.costs.name == "retail_v3.toml"
        seen[row_id] = pb.stable_json_dumps(spec.resolved())
        settings = pb.canonical_settings(LANE, declared["specification"], declared["evaluation"])
        assert settings.fills.model == baseline.row(row_id).fill_model
    assert len(set(seen.values())) == len(seen)
    tranched = json.loads(seen["T21"])["strategy"]["components"][0]
    assert tranched["valid_for_bars"] == 42 and tranched["tranches"] == 21
    assert json.loads(seen["P00"])["strategy"]["components"][0]["valid_for_bars"] == 21
    assert json.loads(seen["D-liq"])["strategy"]["components"][0]["to_weights"] == "decile_neutral_sqrt_dv"


# --------------------------------------------------------------------------- the vintage cache


class FakeFetch:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads, self.calls = payloads, []

    def __call__(self, url: str):
        self.calls.append(url)
        return self.payloads[url], {"Last-Modified": "Wed, 14 Oct 2026 00:00:00 GMT"}


VINTAGE_LINE = "This file was created by using the 202609 CRSP database."


def _french_zip(column: str, index: pd.DatetimeIndex, values: np.ndarray) -> bytes:
    rows = "\n".join(f"{d:%Y%m%d},{v * 100:.4f}" for d, v in zip(index, values))
    text = f"{VINTAGE_LINE}\n\n,{column}\n{rows}\n\nCopyright\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", text)
    return buffer.getvalue()


def test_the_vintage_cache_refuses_a_changed_vintage_and_never_overwrites(tmp_path) -> None:
    url = "https://example.invalid/F-F_Momentum_Factor_daily_CSV.zip"
    good = _french_zip("Mom", pd.bdate_range("2020-01-01", periods=5), np.zeros(5))
    digest = hashlib.sha256(good).hexdigest()
    cache = tmp_path / "vintages"
    fetch = FakeFetch({url: good})
    path = pb.cached_vintage(url, digest, cache, fetch=fetch)
    assert path == cache / digest[:12] / "F-F_Momentum_Factor_daily_CSV.zip" and path.read_bytes() == good
    assert json.loads(path.with_name(path.name + ".json").read_text())["sha256"] == digest
    assert pb.cached_vintage(url, digest, cache, fetch=None) == path and len(fetch.calls) == 1

    changed = FakeFetch({url: good + b"x"})
    other = "1" * 64
    with pytest.raises(pb.BaselineError, match="vintage has changed"):
        pb.cached_vintage(url, other, cache, fetch=changed)
    assert not (cache / other[:12]).exists(), "a refused download writes nothing"
    with pytest.raises(pb.BaselineError, match="fetching is off"):
        pb.cached_vintage(url, other, cache, fetch=None)

    path.write_bytes(b"tampered")
    with pytest.raises(pb.BaselineError, match="never overwritten"):
        pb.cached_vintage(url, digest, cache, fetch=fetch)
    assert path.read_bytes() == b"tampered"


def test_pin_reports_what_a_registration_pins_and_computes_no_statistic(tmp_path) -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    payloads = {pb.FRENCH_URLS["ff_mom_daily"]: _french_zip("Mom", index, np.zeros(5)),
                pb.FRENCH_URLS["ff3_daily"]: _french_zip("RF", index, np.zeros(5))}
    pinned = pb.pin(tmp_path / "vintages", fetch=FakeFetch(payloads))
    for source_id, url in pb.FRENCH_URLS.items():
        entry = pinned[source_id]
        assert entry["sha256"] == hashlib.sha256(payloads[url]).hexdigest() and entry["bytes"] == len(payloads[url])
        assert entry["vintage_line"].startswith("This file was created by using the 202609 CRSP database")
        assert set(entry) == {"url", "sha256", "bytes", "last_modified", "vintage_line", "path"}


# --------------------------------------------------------------------------- statistics on fixtures


def test_the_governing_hurdle_is_the_etf_case_at_the_larger_basis() -> None:
    criteria = tomllib.loads(planned())["criteria"]
    out = pb.hurdles(rf=0.0421, vol_target=0.12, vol_realized=0.127, long_over_equity=0.53931, sleeve=0.44069,
                     criteria=criteria, per_year=252)
    governing = out["governing"]
    assert governing["governing_basis"] == "target"  # a positive return hurdle is larger at the smaller volatility
    expected = tbill_hurdle(0.0421, 0.0, 0.53931, 0.12, sleeve_fraction=0.44069, sleeve_yield=0.0421 - 0.001)
    assert governing["sharpe_annual"] == pytest.approx(expected.sharpe_annual, rel=1e-15)
    assert out["cash_yield_0"]["sharpe_annual"] == pytest.approx(0.0421 / 0.12)
    assert out["cash_yield_0.033"]["sharpe_annual"] == pytest.approx((0.0421 - 0.033 * (1 - 0.53931)) / 0.12)


def test_economics_reads_figure_against_every_case() -> None:
    baseline = parse(planned())
    index = pd.bdate_range("2023-01-02", periods=500)
    rng = np.random.default_rng(3)
    net = pd.Series(rng.normal(0.0002, 0.12 / np.sqrt(252), size=500), index=index)
    cost = pd.Series(0.00001, index=index)
    exposure = pd.DataFrame({"long_over_equity": np.full(500, 0.54)}, index=index)
    econ = pb.economics_of(net, cost, exposure, rho=0.9, prior=0.45, rf=0.0421, baseline=baseline)
    vol = net.std(ddof=1) * np.sqrt(252)
    assert econ["cost_drag_sharpe"] == pytest.approx(0.00001 * 252 / vol)
    assert econ["figure_f3"] == pytest.approx(0.9 * 0.45 - econ["cost_drag_sharpe"])
    assert econ["sleeve_fraction_mean"] == pytest.approx(0.98 - 0.54)
    for case in econ["cases"].values():
        assert case["pass"] == (econ["figure_f3"] > case["sharpe_annual"])
        assert case["excess_over_bills_annual"] == pytest.approx(
            econ["figure_f3"] * case["vol_annual"] - case["return_hurdle_annual"])
    assert econ["pass"] == econ["cases"]["governing"]["pass"]


def test_the_effective_count_benchmark_matches_the_library_at_whole_counts() -> None:
    sharpes = list(np.random.default_rng(1).normal(0.03, 0.01, size=30))
    assert pb.expected_max_sharpe(sharpes, 30) == pytest.approx(expected_max_sharpe_with_n(sharpes, 30), rel=1e-12)
    assert pb.expected_max_sharpe(sharpes, 4.5) < pb.expected_max_sharpe(sharpes, 30)
    assert np.isnan(pb.expected_max_sharpe(sharpes, 1.0))


def test_wash_sale_pairs_count_opposite_trades_within_thirty_days() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-10", "2024-01-25", "2024-03-15", "2024-03-20"])
    trades = pd.DataFrame({"A": [1000.0, -1000.0, 1000.0, 0.0, -1000.0], "B": [np.nan, -500.0, np.nan, 500.0, 0.0]},
                          index=index)
    out = pb.wash_sale_pairs(trades)
    # A: buy 01-02, sell 01-10 (8 days), buy 01-25 (15 days), sell 03-20 (55 days): two pairs, one sell-then-buy.
    # B: sell 01-10, buy 03-15 (65 days): none.
    assert out["pairs"] == 2 and out["sell_then_buy"] == 1


def test_flow_shape_and_same_trade_fraction() -> None:
    even = pb.flow_shape(pd.Series(np.ones(42)), 21)
    assert even["hhi_ratio"] == pytest.approx(1.0) and even["peak_to_mean"] == pytest.approx(1.0)
    lumpy = pb.flow_shape(pd.Series([21.0 if i % 21 == 0 else 0.0 for i in range(42)]), 21)
    assert lumpy["hhi_ratio"] == pytest.approx(21.0) and lumpy["busiest_cycle_share"] == pytest.approx(1.0)
    frame = pd.DataFrame({"bar": ["2024-01-02", "2024-01-02"], "sid": ["A", "B"], "qty": [10.0, -5.0],
                          "notional": [1000.0, -500.0], "equity_prior": [1e5, 1e5]})
    shifted = frame.assign(bar=["2024-01-02", "2024-01-03"])
    out = pb.same_trade_fraction({"x": frame, "y": shifted})
    assert out["pairs"] == 2 and out["mean"] == pytest.approx(1000 / 1500)  # only A matches, on both sides


def _fake_simulation(equity_close: list[float]) -> SimpleNamespace:
    bars = ["2024-01-02", "2024-01-03", "2024-01-04"]
    ledger = pd.DataFrame({"equity_close": equity_close, "trade_cost": [0.0, 1.0, 0.5],
                           "holding_cost": [0.0, 0.1, 0.1]}, index=bars)
    shares = pd.DataFrame({"A": [np.nan, 10.0, np.nan]}, index=bars)
    marks = pd.DataFrame({"A": [np.nan, 101.0, np.nan]}, index=bars)
    fills = pd.DataFrame({"bar": ["2024-01-03", "2024-01-04"], "sid": ["A", "A"], "qty": [10.0, -10.0],
                          "fill_price": [100.0, 103.0]})
    return SimpleNamespace(ledger=ledger, shares=shares, marks=marks, fills=fills,
                           events=pd.DataFrame(columns=["effective_date", "equity_delta"]))


def test_the_n4_identity_holds_through_an_open_and_a_close_and_catches_a_leak() -> None:
    # Buy 10 at 100, mark 101 (+10), sell 10 at 103 (+20), less costs 1.1 then 0.6.
    good = pb.n4_check(_fake_simulation([1000.0, 1000.0 + 10.0 - 1.1, 1000.0 + 10.0 - 1.1 + 20.0 - 0.6]))
    assert good["pass"] and good["max_abs_residual"] == pytest.approx(0.0, abs=1e-9)
    leak = pb.n4_check(_fake_simulation([1000.0, 1000.0 + 10.0 - 1.1, 1000.0 + 10.0 - 1.1 + 20.0 - 0.6 + 0.01]))
    assert not leak["pass"] and leak["worst_bar"] == "2024-01-04"


def test_session_dates_cut_engine_stamps_to_the_session() -> None:
    aware = pd.DatetimeIndex(["2024-03-08", "2024-03-11"]).tz_localize("America/New_York")
    assert list(pb._session_dates(aware)) == list(pd.to_datetime(["2024-03-08", "2024-03-11"]))
    text = ["2024-03-08 00:00:00-05:00", "2024-03-11 00:00:00-04:00"]
    assert list(pb._session_dates(text)) == list(pd.to_datetime(["2024-03-08", "2024-03-11"]))
    with pytest.raises(pb.BaselineError, match="session label"):
        pb._session_dates(["2024-03-08 13:00:00+00:00"])


# --------------------------------------------------------------------------- end to end on synthetic data

ANCHOR = dt.date(2025, 3, 3)
LAST = dt.date(2025, 7, 15)

SMALL = planned(**{
    "trial_budget = 30": "trial_budget = 11",
    "lookback_sessions = 126": "lookback_sessions = 10",
    "refresh_sessions = 21": "refresh_sessions = 4",
    "min_valid_returns = 126": "min_valid_returns = 10",
    "lookback = 252, skip = 21": "lookback = 40, skip = 2",
    "decile = 0.10, max_symbol_abs_weight = 0.10": "decile = 0.25, max_symbol_abs_weight = 0.35",
    "folds = { test_sessions = 63, formation_sessions = 312 }":
        "folds = { test_sessions = 40, formation_sessions = 270 }",
    "p00_anchor = \"2022-03-29\"": f"p00_anchor = \"{ANCHOR}\"",
    "scored_last = \"2026-06-12\"": f"scored_last = \"{LAST}\"",
    "parity_series = \"results/2026-09-vol-momentum/series.csv\"": "parity_series = \"parity.csv\"",
    "phases = { ids = \"P00..P20\", tranches = 1, offsets = \"0..20\" }":
        "phases = { ids = \"P00..P03\", tranches = 1, offsets = \"0..3\" }",
    "T21 = { tranches = 21 }": "T21 = { tranches = 4 }",
    "D-split = { tranches = 21,": "D-split = { tranches = 4,",
    "D-liq = { tranches = 21,": "D-liq = { tranches = 4,",
    "D-part = { tranches = 21, keep_fraction = 0.8, seeds = [11, 22, 33] }":
        "D-part = { tranches = 4, keep_fraction = 0.8, seeds = [11] }",
    "T21-25k = { tranches = 21,": "T21-25k = { tranches = 4,",
    "T21-1m = { tranches = 21,": "T21-1m = { tranches = 4,",
    "T3-25k = { tranches = 3, offsets = \"r, r+7, r+14\",": "T3-25k = { tranches = 2, offsets = \"r, r+2\",",
    "block_bars = 42\nn_resamples = 2000": "block_bars = 21\nn_resamples = 200",
    "\"P00:1/21\"": "\"P00:1/4\"",
    "fill_model = \"full_next_open\"\n": f"fill_model = \"full_next_open\"\nuniverse_index = \"{EQUITY_INDEX}\"\n",
})
SMALL_EXTRA = """
[implementation.eligibility]
min_price = 0.0
min_median_dollar_volume = 1.0
return_history_bars = 40
"""


def _git(root: Path, *args: str) -> None:
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    subprocess.run(["git", "-C", str(root), *args], env=env, check=True, capture_output=True)


def _snapshot(*roots: Path) -> dict[str, str]:
    """Every regular file under the roots with its sha256, not following symlinks and skipping .git."""
    out = {}
    for root in roots:
        for directory, subdirectories, files in os.walk(root, followlinks=False):
            subdirectories[:] = [name for name in subdirectories if name != ".git"]
            for name in files:
                path = Path(directory) / name
                if path.is_file() and not path.is_symlink():
                    out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("public_baseline")
    store = base / "store" / "dataset"
    manifest = write_synthetic_dataset(store, seed=7, scenarios="all", n_equities=12, n_etfs=2,
                                       start=dt.date(2024, 1, 2), end=dt.date(2025, 8, 29))
    # The engine routes a synthetic manifest to the engineering channel; the driver's registered path is the
    # point of this test, so the copy declares itself market data (its content is still synthetic).
    manifest = dataclasses.replace(manifest, provenance="market")
    manifest.write(store / "manifest.json")
    rng = np.random.default_rng(10)
    long_index = pd.bdate_range("1985-01-02", "2026-07-31")
    payloads = {pb.FRENCH_URLS["ff_mom_daily"]: _french_zip("Mom", long_index, rng.normal(0.0004, 0.009,
                                                                                            len(long_index))),
                pb.FRENCH_URLS["ff3_daily"]: _french_zip("RF", long_index, np.full(len(long_index), 0.00015))}
    registration = SMALL.replace(f"manifest_id = \"{MANIFEST_ID}\"", f"manifest_id = \"{manifest.id}\"")
    registration = registration.replace("vintage = \"demotion_b1_2026-06-16\"", f"vintage = \"{manifest.vintage}\"")
    registration = registration.replace("path = \"data/frozen/demotion_b1\"", f"path = \"{store}\"")
    for prefix, url in (("momentum", pb.FRENCH_URLS["ff_mom_daily"]), ("cash", pb.FRENCH_URLS["ff3_daily"])):
        registration = registration.replace(f"{prefix}_sha256 = \"{'0' * 64}\"",
                                            f"{prefix}_sha256 = \"{hashlib.sha256(payloads[url]).hexdigest()}\"", 1)
    momentum = payloads[pb.FRENCH_URLS["ff_mom_daily"]]
    registration = registration.replace("momentum_bytes = 1\n", f"momentum_bytes = {len(momentum)}\n")
    registration = registration.replace("momentum_vintage_line = \"placeholder\"",
                                        f"momentum_vintage_line = \"{VINTAGE_LINE}\"")
    registration = registration.replace("\n[guards]", SMALL_EXTRA + "\n[guards]")
    return SimpleNamespace(base=base, store=store, manifest=manifest, payloads=payloads, registration=registration)


#: The registered stack, which every synthetic run here is charged under, square-root term included.
REGISTERED_STACK = "examples/reference/costs/retail_v3.toml"

#: The tracked files a row specification and its charged policy read, copied into each test checkout.
INPUTS = (REGISTERED_STACK, *pb.DEFAULT_POLICY.values())


def _repo(world, name: str) -> Path:
    root = world.base / name
    (root / "registrations").mkdir(parents=True)
    (root / "docs").mkdir()
    for relative in INPUTS:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes((LANE / relative).read_bytes())
    window = sessions.sessions_between(dt.date(2025, 6, 2), LAST)
    rng = np.random.default_rng(5)
    pd.DataFrame({"in_house|net": rng.normal(0.0, 0.01, len(window))},
                 index=pd.Index([day.isoformat() for day in window])).to_csv(root / "parity.csv")
    (root / pb.REGISTRATION).write_text(world.registration, encoding="utf-8")
    (root / pb.DESIGN_DOC).write_text("# synthetic design\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "register")
    return root


def _main(root: Path, *args: str, fetch=None) -> int:
    return pb.main(["--workspace", str(root), *args], fetch=fetch)


def _ledger(root: Path) -> list[dict]:
    return list(TrialLog(root / "trials" / f"{pb.SELECTION_SET}.jsonl").verify())


def test_the_driver_refuses_before_its_registration_is_committed(world, capsys) -> None:
    root = _repo(world, "uncommitted")
    (root / pb.REGISTRATION).write_text((root / pb.REGISTRATION).read_text() + "\n# edited\n", encoding="utf-8")
    assert _main(root, "--part", "rows", fetch=FakeFetch(world.payloads)) == 2
    assert "differs from HEAD" in capsys.readouterr().err
    assert not (root / "trials").exists() and not (root / pb.OUT_DIR).exists()


def test_the_driver_refuses_a_french_file_that_is_not_the_pinned_vintage(world, capsys) -> None:
    root = _repo(world, "vintage")
    changed = {url: payload + b"\n" for url, payload in world.payloads.items()}
    assert _main(root, "--part", "rows", fetch=FakeFetch(changed)) == 2
    assert "vintage has changed" in capsys.readouterr().err
    assert not (root / "trials").exists()


def test_the_driver_refuses_an_input_file_edited_after_the_registration(world, capsys) -> None:
    root = _repo(world, "edited_stack")
    stack = root / REGISTERED_STACK
    stack.write_text(stack.read_text() + "\n# edited\n", encoding="utf-8")
    assert _main(root, "--part", "rows", fetch=FakeFetch(world.payloads)) == 2
    assert f"{REGISTERED_STACK} differs from HEAD" in capsys.readouterr().err
    assert not (root / "trials").exists()


def test_check_validates_without_evaluating(world) -> None:
    root = _repo(world, "check")
    report = pb.check(root, SimpleNamespace(dataset=None))
    assert report["rows"][:2] == ["P00", "T21"] and len(report["specifications"]) == 11
    assert report["dataset"]["manifest_id"] == world.manifest.id and report["reference_cached"] == {
        "momentum": False, "cash": False}
    assert not (root / "trials").exists() and not (root / pb.OUT_DIR).exists()


def test_a_partition_filters_membership_and_declares_itself(world) -> None:
    base = Dataset.open(world.store)
    part = pb.PartitionedDataset(base, index_id=EQUITY_INDEX, seed=11, keep=0.5)
    assert part.manifest.id != base.manifest.id and part.manifest.files == base.manifest.files
    frame = part.read_table("membership.parquet")
    members = set(base.read_table("membership.parquet").query("index_id == @EQUITY_INDEX")["sid"].astype(str))
    kept = set(frame.loc[frame["index_id"] == EQUITY_INDEX, "sid"].astype(str))
    assert kept == set(universe_partition.partition(sorted(members), 11, 0.5)) and kept < members
    assert set(part.kept) == kept and set(part.dropped) == members - kept


def test_end_to_end_one_row_per_row_failures_recorded_parity_gate_and_declared_writes(world, monkeypatch) -> None:
    root = _repo(world, "e2e")
    fetch = FakeFetch(world.payloads)
    parity = root / "parity.csv"
    window = sessions.sessions_between(dt.date(2025, 6, 2), LAST)
    before = _snapshot(world.store)
    workspace_before = _snapshot(root)

    # 1. P00 runs first and fails its parity gate: the run stops, nothing else is evaluated.
    assert _main(root, "--part", "rows", fetch=fetch) == 3
    rows = _ledger(root)
    assert [(row["arm_id"], row["event"], row["kind"]) for row in rows] == [("P00", "started", "new"),
                                                                            ("P00", "finished", "new")]
    gate = json.loads((root / pb.OUT_DIR / "parity.json").read_text())
    assert gate["pass"] is False and gate["n"] == len(window)
    summary = json.loads((root / pb.OUT_DIR / "summary.json").read_text())
    assert summary["verdict"] == "mechanics_failure" and summary["approved"] is False
    assert json.loads((root / pb.OUT_DIR / "rows" / "P00" / "row.json").read_text())["mechanics"]["pass"] is False
    assert _main(root, "--part", "rows", "--rows", "T21", fetch=fetch) == 3, "no row runs behind a failed gate"

    # The fault "is diagnosed": here the recorded series is replaced by what P00 produces.
    p00 = pd.read_csv(root / pb.OUT_DIR / "rows" / "P00" / "series.csv", index_col="date")["net"]
    pd.DataFrame({"in_house|net": p00.loc[[day.isoformat() for day in window]].to_numpy()},
                 index=pd.Index([day.isoformat() for day in window])).to_csv(parity)
    assert _main(root, "--part", "rows", "--retry", "P00", fetch=fetch) == 2, "an uncommitted input is refused"
    _git(root, "commit", "-q", "-am", "diagnose the parity series")

    # 2. A row whose specification cannot be built records a failed row itself, and the run stops there.
    real_load = pb.load_spec

    def refuse_d_liq(path, **kwargs):
        if "D-liq" in str(path):
            raise ValueError("synthetic refusal")
        return real_load(path, **kwargs)

    monkeypatch.setattr(pb, "load_spec", refuse_d_liq)
    assert _main(root, "--part", "rows", "--retry", "P00", fetch=fetch) == 2
    monkeypatch.undo()
    rows = _ledger(root)
    assert [(row["arm_id"], row["event"]) for row in rows if row["arm_id"] == "D-liq"] == [("D-liq", "failed")]
    assert [row["kind"] for row in rows if row["arm_id"] == "P00"] == ["new", "new", "retry", "retry"]
    assert json.loads((root / pb.OUT_DIR / "parity.json").read_text())["pass"] is True
    assert any(path.name.startswith("P00.superseded-") for path in (root / pb.OUT_DIR / "rows").iterdir())

    # 3. A failure inside the engine is recorded by the engine: a started row and a failed row.
    import prism.evaluate as engine

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic engine fault")

    monkeypatch.setattr(engine, "simulate", broken)
    with pytest.raises(RuntimeError, match="synthetic engine fault"):
        _main(root, "--part", "rows", "--retry", "failed", fetch=fetch)
    monkeypatch.undo()
    assert [(row["event"], row["kind"]) for row in _ledger(root) if row["arm_id"] == "D-liq"] == [
        ("failed", "new"), ("started", "retry"), ("failed", "retry")]

    # 4. Everything left runs; the summary is written once every row has finished.
    assert _main(root, "--part", "rows", "--retry", "failed", fetch=fetch) == 0
    rows = _ledger(root)
    baseline = pb.load_baseline(root)
    for row in baseline.rows:
        events = [(entry["event"], entry["kind"]) for entry in rows if entry["arm_id"] == row.id]
        new = [event for event in events if event[1] == "new"]
        assert ("finished", "new") in new or ("failed", "new") in new, row.id
        assert sum(1 for event in events if event[0] == "started" and event[1] == "new") <= 1, row.id
        assert events[-1][0] == "finished", row.id
    assert {row["selection_set_id"] for row in rows} == {pb.SELECTION_SET}

    out = root / pb.OUT_DIR
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["criteria"]) == {"publication_bar", "tracking", "economics", "no_contradiction", "mechanics",
                                        "tranching"}
    assert summary["approved"] == all(block["pass"] for block in summary["criteria"].values())
    assert summary["criteria"]["mechanics"]["parity"]["pass"] is True
    assert summary["criteria"]["mechanics"]["candidate"]["n4"]["pass"] is True
    assert summary["criteria"]["mechanics"]["candidate"]["target_gross_within_ceiling"] is True
    assert set(summary["criteria"]["economics"]["cases"]) == {"governing", "cash_yield_0", "cash_yield_0.033"}
    assert summary["deflation"]["n_rows"] == 11 and 2.0 <= summary["deflation"]["n_effective"] <= 11.0
    assert set(summary["breadth"]["ic_profile"]) == {str(h) for h in pb.IC_HORIZONS}
    assert set(summary["rounding"]) == {"T21", "T21-25k", "T21-1m", "T3-25k"}
    assert set(summary["diagnostics"]) == {"D-split", "D-liq", "D-part1", "holdings_diversity"}
    assert len(summary["phase_dispersion"]["per_phase"]) == 4
    for row in baseline.rows:
        record = json.loads((out / "rows" / row.id / "row.json").read_text())
        assert record["mechanics"]["n4"]["pass"], row.id
        assert record["specification_sha256"] == hashlib.sha256(
            pb.stable_json_dumps(pb.row_specification(baseline, row)).encode()).hexdigest()
    fills = pd.read_parquet(out / "rows" / "D-split" / "fills.parquet")
    assert set(fills["leg"]) == {"open", "close"}
    assert summary["diagnostics"]["D-split"]["executable_return"] == "split_open_close_fill_marked_close_to_close"
    assert {summary["conventions"]["executable_return"][row.id] for row in baseline.rows if row.id != "D-split"} == {
        "next_open_fill_marked_close_to_close"}
    assert all((out / name).is_file() for name in ("series.csv", "phase_dispersion.json", "flow.json"))

    # Every attempt is listed beside the verdict, and a finished row is never re-run as a retry.
    attempts = summary["attempts"]
    assert [(a["kind"], a["event"], a["current"]) for a in attempts["P00"]] == [("new", "finished", False),
                                                                                ("retry", "finished", True)]
    first_p00 = attempts["P00"][0]
    assert first_p00["mechanics_pass"] is False and first_p00["path"].startswith("rows/P00.superseded-")
    assert [(a["kind"], a["event"]) for a in attempts["D-liq"]] == [("new", "failed"), ("retry", "failed"),
                                                                   ("retry", "finished")]
    assert all(a["record_sha256"] for row_attempts in attempts.values() for a in row_attempts if a["current"])
    count = len(_ledger(root))
    assert _main(root, "--part", "rows", "--retry", "T21", fetch=fetch) == 2
    assert _main(root, "--part", "rows", "--retry", "P00", fetch=fetch) == 2, "P00's gate passed"
    assert len(_ledger(root)) == count

    # 5. Capacity re-prices the recorded trades and appends no ledger row.
    count = len(_ledger(root))
    assert _main(root, "--part", "capacity", fetch=fetch) == 0
    assert len(_ledger(root)) == count
    capacity = json.loads((out / "capacity" / "summary.json").read_text())
    assert set(capacity["rows"]) == {"T21", "P00", "D-split", "D-liq"}
    table = pd.read_csv(out / "capacity" / "P00.csv")
    assert set(table["synchronized_fraction"].round(6)) == {1.0, 0.25}
    assert len(table) == 4 * 2 * 3 * 6

    # 6. Nothing was written outside the declared places, and the dataset was only read.
    assert _snapshot(world.store) == before
    declared = [root / pb.OUT_DIR, root / "trials", root / "state", root / "runs", root / "logs",
                root / "data" / "public" / "vintages"]
    for path, digest in _snapshot(root).items():
        if workspace_before.get(path) == digest:
            continue
        candidate = Path(path)
        in_results_run = candidate.parent.parent == root / "results" and candidate.parent.name.startswith("run-")
        assert in_results_run or any(place in candidate.parents for place in declared) or path == str(parity), path


@pytest.mark.parametrize("row_id", ["P00", "D-split"])
def test_the_registered_cost_stack_charges_its_participation_term_as_the_capacity_repricing_does(
    world, row_id
) -> None:
    """retail_v3's square-root term, charged by the engine on each decided order at its fill session's volume.

    The participation dollars the simulation charged on a bar equal
    ``prism.validation.capacity.crowd_impact_cost`` with no other deployment trading (F = 0), at
    Y * sigma = participation_coeff / 1e4, one full-day window and the stack's floor, on each order's
    decided notional against close times volume of the session it filled in. That is the identity design
    section 9's re-pricing of recorded trades rests on. D-split's two legs together are one order's charge.
    """
    root = _repo(world, f"retail_v3_{row_id}")
    baseline = pb.load_baseline(root)
    stack_path = baseline.table("implementation")["cost_stack"]
    assert stack_path == "examples/reference/costs/retail_v3.toml"
    stack = tomllib.loads((root / stack_path).read_text(encoding="utf-8"))
    coeff, floor = stack["impact"]["participation_coeff"], stack["dollar_volume"]["impact"]["floor_dollars"]
    assert coeff == 126.0 and stack["dollar_volume"]["impact"]["estimator"] == "same_bar"
    dataset = pb.open_dataset(world.store, baseline)
    with pb.observe_simulation() as seen:
        record = pb.evaluate_row(root=root, out=root / pb.OUT_DIR, baseline=baseline, row=baseline.row(row_id),
                                 dataset=dataset, folds=pb.build_fold_table(baseline), kind="new",
                                 code_commit="test")
    assert record["mechanics"]["n4"]["pass"], record["mechanics"]["n4"]
    (simulation,) = seen

    # One cell per completely filled order, at the session it filled in (both legs of a D-split order share it).
    orders = {order.client_order_id: order for order in simulation.orders}
    complete = {cid for cid, order in orders.items() if order.status == "filled"}
    fills = simulation.fills
    clean = [bar for bar, ids in fills.groupby("bar")["client_order_id"].agg(set).items() if ids <= complete]
    assert len(clean) >= 10, "too few bars on which every fill belongs to a completed order"
    close, volume = pb._price_panels(dataset, sorted(set(fills["sid"].astype(str))), str(max(fills["bar"])))
    dollar = close * volume
    notional, liquidity = {}, {}
    for cid in sorted(complete):
        order = orders[cid]
        (bar,) = {fill.bar for fill in order.fills}
        if bar not in clean:
            continue
        notional[(bar, cid)] = abs(order.intent.qty * order.intent.reference_price)
        liquidity[(bar, cid)] = float(dollar.at[pd.Timestamp(bar), order.sid])
    trades = pd.Series(notional).unstack().fillna(0.0)
    volumes = pd.Series(liquidity).unstack().reindex(index=trades.index, columns=trades.columns)
    priced = crowd_impact_cost(trades, volumes.fillna(1.0), 1.0, 1.0,
                               CrowdScenario(impact_y=coeff / 1e4, windows=((1.0, 1.0),)), floor_dollars=floor)
    expected = priced.sum(axis=1)
    charged = simulation.costs["impact_participation"].reindex(expected.index)
    assert float(expected.sum()) > 0.0
    np.testing.assert_allclose(charged.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12)
    if row_id == "D-split":
        legs = fills.loc[fills["client_order_id"].isin(complete) & fills["bar"].isin(clean)]
        assert set(legs["execution_id"].astype(str).str.endswith(":close")) == {True, False}
