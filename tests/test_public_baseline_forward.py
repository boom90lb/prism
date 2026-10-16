"""Synthetic-data tests for the forward stop-rule reader (``research/scripts/public_baseline_forward.py``).

Nothing here reads market data. The clock is checked on the committed registration
(its dates and thresholds only); the stop logic and the paper-stream corrections on
constructed series; the run and read steps end to end on a synthetic dataset, fake
French files and a scaled-down registration of the same shape.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.io import sessions
from prism.io.synthetic import write_synthetic_dataset
from prism.live import ledgers
from prism.live.loop import append_equity_ledger, append_targets_ledger
from prism.live.panel_log import record_panel
from prism.live.state import ComponentState, LoopState, StateStore
from prism.live.vol_sizing import SIZING_LEDGER_FILENAME
from prism.validation.trials import TrialLog
from research.scripts import public_baseline as pb
from research.scripts import public_baseline_forward as fwd
from tests.test_public_baseline import (
    INPUTS,
    SMALL,
    SMALL_EXTRA,
    VINTAGE_LINE,
    FakeFetch,
    _french_zip,
    _git,
    _snapshot,
)

LANE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registered() -> pb.Baseline:
    """The committed registration, parsed (no data is read)."""
    return pb.load_baseline(LANE)


# --------------------------------------------------------------------------- the clock


def test_the_registered_clock_starts_after_the_registration_and_reads_every_63_sessions(registered) -> None:
    clock = fwd.build_clock(registered)
    assert clock.registered == dt.date(2026, 10, 16) and clock.first_forward == dt.date(2026, 10, 19)
    # Three whole 63-session windows cover lookback 126 + refresh 21 = 147 sessions.
    assert clock.warmup_sessions == 189 and clock.test_sessions == 63 and clock.formation_sessions == 312
    assert [(r.name, r.threshold) for r in clock.reads] == [("read_1", 126), ("read_2", 189), ("read_3", 252)]
    for read in clock.reads:
        held = sessions.sessions_between(clock.first_forward, sessions.previous_session(read.asof))
        assert len(held) == read.threshold, read.name
    assert [str(r.asof) for r in clock.reads] == ["2027-04-21", "2027-07-22", "2027-10-20"]
    # A fourth read (315 sessions) falls past the bundled session table and is not scheduled.
    with pytest.raises(sessions.SessionTableRangeError):
        fwd.shift_sessions(clock.first_forward, 315)


def test_a_read_is_refused_before_its_scheduled_asof_and_taken_at_it(registered) -> None:
    clock = fwd.build_clock(registered)
    for early in (dt.date(2026, 10, 16), dt.date(2026, 12, 1), dt.date(2027, 4, 20)):
        with pytest.raises(fwd.ReadRefused, match="before the first scheduled read"):
            fwd.due_read(clock, early)
    assert fwd.due_read(clock, dt.date(2027, 4, 21)).name == "read_1"
    assert fwd.due_read(clock, dt.date(2027, 7, 21)).name == "read_1"
    assert fwd.due_read(clock, dt.date(2027, 7, 22)).name == "read_2"


def test_the_forward_fold_table_is_the_registered_geometry_from_the_warm_up(registered) -> None:
    clock = fwd.build_clock(registered)
    table = fwd.forward_fold_table(clock, clock.reads[0])
    assert table.n_folds == 5
    starts = [fold.test_start.date() for fold in table.folds]
    assert starts[3] == clock.first_forward, "the first forward window opens on the first forward session"
    for fold in table.folds:
        tested = sessions.sessions_between(fold.test_start.date(), fold.test_end.date())
        formed = sessions.sessions_between(fold.formation_start.date(), fold.formation_end.date())
        assert len(tested) == 63 and len(formed) == 312
        assert sessions.next_session(fold.formation_end.date()) == fold.test_start.date()
    assert table.folds[-1].test_end.date() == sessions.previous_session(clock.reads[0].asof)


def test_the_command_line_refuses_before_the_first_read_and_after_today(capsys) -> None:
    assert fwd.main(["status", "--workspace", str(LANE), "--asof", "2026-11-02"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["due"] is None and "first possible at as-of 2027-04-21" in report["refusal"]
    for command in ("freeze", "import", "run", "read"):
        assert fwd.main([command, "--workspace", str(LANE), "--asof", "2026-11-02"],
                        today=dt.date(2026, 11, 2)) == 2
        assert "before the first scheduled read" in capsys.readouterr().err
    assert fwd.main(["read", "--workspace", str(LANE), "--asof", "2027-04-21"], today=dt.date(2026, 11, 2)) == 2
    assert "after today" in capsys.readouterr().err


# --------------------------------------------------------------------------- the stop logic


def _streams(n: int = 200, seed: int = 1):
    index = pd.bdate_range("2027-01-04", periods=n)
    rng = np.random.default_rng(seed)
    reference = pd.Series(rng.normal(0.0003, 0.008, n), index=index)
    tracked = 0.9 * reference + pd.Series(rng.normal(0.0004, 0.003, n), index=index)
    unrelated = pd.Series(rng.normal(0.0004, 0.008, n), index=index)
    return reference, tracked, unrelated


def _conditions(registered, net, reference, *, hurdle=0.2, previous=None, excess=None):
    return fwd.stream_conditions(net, reference=reference, post_publication_sharpe=0.465, cost_drag=0.05,
                                 governing_hurdle=hurdle, previous_d=previous,
                                 excess=net if excess is None else excess, baseline=registered)


def test_a_tracking_book_above_its_hurdle_keeps_eligibility(registered) -> None:
    reference, tracked, _ = _streams()
    read = _conditions(registered, tracked, reference)
    assert not any(read["conditions"].values()) and read["removes_eligibility"] is False
    assert read["c_tracking"]["daily_corr"] > 0.9
    figure = read["c_tracking"]["daily_corr"] * 0.465 - 0.05
    assert read["d_economics"]["figure"] == pytest.approx(figure) and read["d_economics"]["margin"] > 0


def test_each_condition_fires_on_its_own_fixture(registered) -> None:
    reference, tracked, unrelated = _streams()
    losing = tracked - 0.004
    assert _conditions(registered, losing, reference)["conditions"]["a_interval_below_zero"] is True
    crashed = tracked.copy()
    crashed.iloc[100] = -0.30
    read = _conditions(registered, crashed, reference)
    assert read["conditions"]["b_drawdown_above_limit"] is True and read["removes_eligibility"] is True
    assert read["b_max_drawdown_from_forward_peak"] > 0.25
    read = _conditions(registered, unrelated, reference)
    assert read["conditions"]["c_fidelity_below_floor"] is True and read["removes_eligibility"] is True


def test_economics_removes_eligibility_only_on_two_consecutive_reads(registered) -> None:
    reference, tracked, _ = _streams()
    for previous, removed in ((None, False), (False, False), (True, True)):
        read = _conditions(registered, tracked, reference, hurdle=0.5, previous=previous)
        assert read["conditions"]["d_economics_below_hurdle"] is True
        assert read["conditions"]["d_two_consecutive_reads"] is removed
        assert read["removes_eligibility"] is removed, previous


def test_the_excess_read_is_report_only_and_states_its_horizon(registered) -> None:
    reference, tracked, _ = _streams()
    sunk = tracked * 0.0 - 0.01
    read = _conditions(registered, tracked, reference, excess=sunk)
    assert read["e_report_only"]["read"]["verdict"] == "refuted" and read["removes_eligibility"] is False
    horizon = read["e_report_only"]["detection_horizon"]
    figure, margin = read["d_economics"]["figure"], read["d_economics"]["margin"]
    assert horizon["years_to_reject_zero_sharpe_at_figure"] == pytest.approx((1.6448536269514722 / figure) ** 2)
    assert horizon["years_to_detect_margin_over_bills"] == pytest.approx((1.6448536269514722 / margin) ** 2)
    assert "about 29 years" in horizon["registered_planning_statement"]
    below = _conditions(registered, tracked, reference, hurdle=5.0)
    assert below["e_report_only"]["detection_horizon"]["years_to_detect_margin_over_bills"] is None


def test_a_reference_that_barely_overlaps_refuses_the_read(registered) -> None:
    reference, tracked, _ = _streams()
    with pytest.raises(fwd.ReadRefused, match="condition \\(c\\) cannot be read"):
        _conditions(registered, tracked, reference.iloc[:40])


def test_the_forward_excess_credits_the_etf_share_at_rf_less_its_fee(registered) -> None:
    index = pd.bdate_range("2027-01-04", periods=3)
    net = pd.Series([0.001, -0.002, 0.0], index=index)
    long_side = pd.Series([0.5, 1.2, 0.98], index=index)
    out = fwd.forward_excess(net, long_side, rf=0.04, baseline=registered)
    share = np.array([0.48, 0.0, 0.0])
    np.testing.assert_allclose(out.to_numpy(), net.to_numpy() + share * (0.04 - 0.001) / 252 - 0.04 / 252)
    with pytest.raises(fwd.ReadRefused):
        fwd.forward_excess(net, long_side.iloc[:2], rf=0.04, baseline=registered)


# --------------------------------------------------------------------------- the paper stream

S = [d.isoformat() for d in sessions.sessions_between(dt.date(2027, 1, 4), dt.date(2027, 1, 7))]
FILLS = [{"filled_bar": S[0], "symbol": "SGOV", "qty": 400.0}, {"filled_bar": S[0], "symbol": "XYZ", "qty": -50.0},
         {"filled_bar": S[1], "symbol": "XYZ", "qty": -50.0}]
EQUITY = {S[0]: 100_000.0, S[1]: 101_000.0, S[2]: 101_000.0, S[3]: 100_495.0}
CLOSE = pd.DataFrame({"SGOV": [100.0, 100.01, 99.66, 99.67], "XYZ": [20.0, 22.0, 21.0, 21.5]},
                     index=pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") for d in S]))


def _distributions() -> pd.DataFrame:
    return pd.DataFrame({"ex_date": pd.to_datetime([S[2], "2027-01-09"]), "pay_date": pd.to_datetime([S[3], S[3]]),
                         "amount": [0.35, 0.40]})


def _expected_paper() -> list[float]:
    borrow = fwd.PAPER_BORROW_ANNUAL / 252
    return [0.01 - borrow * 50 * 20.0 / 100_000.0,
            0.0 + (0.35 * 400 - borrow * 100 * 22.0) / 101_000.0,
            -0.005 - borrow * 100 * 21.0 / 101_000.0]


def test_paper_returns_get_the_distributions_and_the_borrow_the_paper_account_omits() -> None:
    positions = fwd.positions_by_bar(FILLS)
    assert fwd.held_at(positions, S[0]) == {"SGOV": 400.0, "XYZ": -50.0}
    assert fwd.held_at(positions, S[3]) == {"SGOV": 400.0, "XYZ": -100.0} and fwd.held_at(positions, "2026-12-31") == {}
    series = {"returns": np.array([0.01, 0.0, -0.005]), "bars": S[1:], "prev_bars": S[:3]}
    frame = fwd.correct_paper_returns(series, EQUITY, positions, CLOSE, _distributions(), "SGOV",
                                      borrow_annual=fwd.PAPER_BORROW_ANNUAL, per_year=252)
    np.testing.assert_allclose(frame["net"].to_numpy(), _expected_paper(), rtol=0, atol=1e-15)
    assert frame["distribution"].tolist() == [0.0, 140.0, 0.0]
    unpriced = CLOSE.drop(columns="XYZ")
    with pytest.raises(fwd.ReadRefused, match="no close for the short position XYZ"):
        fwd.correct_paper_returns(series, EQUITY, positions, unpriced, _distributions(), "SGOV",
                                  borrow_annual=fwd.PAPER_BORROW_ANNUAL, per_year=252)


def _paper_run_dir(root: Path, *, positions: dict[str, float], bars=S, equity=EQUITY, close=CLOSE, fills=FILLS,
                   tranches: int = 21, base: int = 13, distributions: pd.DataFrame | None = None,
                   distributions_root: Path | None = None) -> Path:
    """A run directory the tranched, scaled, T-bill-holding paper loop could have written."""
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    components = [ComponentState(component_id=f"decile_neutral/t{j:02d}", phase_offset_bars=(base + j) % tranches)
                  for j in range(tranches)]
    StateStore(run_dir / "state.json").save(LoopState(components=components, positions=positions))
    ledgers.append_rows(run_dir / SIZING_LEDGER_FILENAME, [{"decision_bar": bars[0], "scale": 1.0}])
    weights = pd.Series({"XYZ": -0.01, "SGOV": 0.4})
    append_targets_ledger(run_dir / "targets.jsonl", bars[0], bars[0], weights,
                          pd.Series({"XYZ": 20.0, "SGOV": 100.0}), 1e5,
                          extra={"cash_holding": {"symbol": "SGOV", "weight": 0.4}})
    for bar in bars:
        append_equity_ledger(run_dir / "equity.jsonl", bar, equity[bar], 0.0)
    ledgers.append_rows(run_dir / "fills.jsonl", list(fills))
    record_panel(run_dir, bars[-1], {"close": close})
    folder = (distributions_root or root) / fwd.DISTRIBUTIONS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (_distributions() if distributions is None else distributions).to_csv(folder / "SGOV.csv", index=False,
                                                                           date_format="%Y-%m-%d")
    (folder / "SGOV.provenance.json").write_text("{}")
    return run_dir


def test_the_paper_loader_rebuilds_the_stream_from_its_run_directory(registered, tmp_path) -> None:
    run_dir = _paper_run_dir(tmp_path, positions={"SGOV": 400.0, "XYZ": -100.0})
    found = fwd.load_paper_stream(tmp_path, run_dir, registered, first=dt.date.fromisoformat(S[1]),
                                  last=dt.date.fromisoformat(S[3]))
    assert found["etf"] == "SGOV" and found["panel_bar"] == S[3] and found["n_fills"] == 3
    raw = [EQUITY[b] / EQUITY[p] - 1.0 for p, b in zip(S[:3], S[1:])]
    np.testing.assert_allclose(found["frame"]["raw"].to_numpy(), raw)
    np.testing.assert_allclose(found["frame"]["net"].to_numpy() - found["frame"]["raw"].to_numpy(),
                               np.array(_expected_paper()) - np.array([0.01, 0.0, -0.005]), rtol=0, atol=1e-12)


def test_the_paper_loader_refuses_a_run_directory_that_is_not_the_t21_book(registered, tmp_path) -> None:
    with pytest.raises(fwd.ReadRefused, match="not a T21 paper stream"):
        fwd.load_paper_stream(tmp_path, tmp_path, registered, first=dt.date(2027, 1, 4), last=dt.date(2027, 1, 7))
    run_dir = _paper_run_dir(tmp_path / "drift", positions={"SGOV": 400.0, "XYZ": -90.0})
    with pytest.raises(fwd.ReadRefused, match="differ from state.json"):
        fwd.load_paper_stream(tmp_path / "drift", run_dir, registered, first=dt.date(2027, 1, 5),
                              last=dt.date(2027, 1, 7))


# --------------------------------------------------------------------------- end to end on synthetic data

#: A scaled-down registration of the committed shape: 4-session cycle and tranches, a 10-session volatility
#: window, 20-session test windows after 270-session formation windows, reads at 60 forward sessions then 20.
CREATED = dt.date(2025, 3, 3)
FORWARD = SMALL.replace("created = 2026-10-16", f"created = {CREATED}").replace(
    "folds = { test_sessions = 40, formation_sessions = 270 }",
    "folds = { test_sessions = 20, formation_sessions = 270 }").replace(
    "first_read_min_sessions = 126\nread_every_sessions = 63",
    "first_read_min_sessions = 60\nread_every_sessions = 20")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("public_baseline_forward")
    store = base / "store" / "dataset"
    manifest = write_synthetic_dataset(store, seed=11, scenarios="all", n_equities=12, n_etfs=2,
                                       start=dt.date(2024, 1, 2), end=dt.date(2025, 8, 29))
    # The engine routes a synthetic manifest to the engineering channel; the reader's registered path is the
    # point of this test, so the copy declares itself market data (its content is still synthetic).
    manifest = dataclasses.replace(manifest, provenance="market")
    manifest.write(store / "manifest.json")
    rng = np.random.default_rng(12)
    index = pd.bdate_range("1985-01-02", "2025-07-31")
    payloads = {pb.FRENCH_URLS["ff_mom_daily"]: _french_zip("Mom", index, rng.normal(0.0004, 0.009, len(index))),
                pb.FRENCH_URLS["ff3_daily"]: _french_zip("RF", index, np.full(len(index), 0.00015))}
    registration = FORWARD.replace(f"vintage = \"demotion_b1_2026-06-16\"", f"vintage = \"{manifest.vintage}\"")
    registration = registration.replace("\n[guards]", SMALL_EXTRA + "\n[guards]")
    return {"base": base, "store": store, "payloads": payloads, "registration": registration}


def _repo(world, name: str) -> Path:
    root = world["base"] / name
    for relative in (*INPUTS, str(fwd.READER)):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes((LANE / relative).read_bytes())
    (root / "parity.csv").write_text("date,in_house|net\n2025-01-02,0.0\n")
    (root / pb.REGISTRATION).parent.mkdir(parents=True)
    (root / pb.REGISTRATION).write_text(world["registration"], encoding="utf-8")
    (root / pb.DESIGN_DOC).parent.mkdir(parents=True)
    (root / pb.DESIGN_DOC).write_text("# synthetic design\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "register")
    return root


def test_end_to_end_run_then_read_in_order_once_and_inside_declared_places(world, capsys) -> None:
    root = _repo(world, "e2e")
    fetch = FakeFetch(world["payloads"])
    clock = fwd.build_clock(pb.load_baseline(root))
    first, second = clock.reads[0], clock.reads[1]
    assert clock.first_forward == dt.date(2025, 3, 4) and clock.warmup_sessions == 20 and first.threshold == 60
    # A paper run directory holding the scaled-down T21 book (4 tranches on a 4-session cycle) over the window.
    days = sessions.sessions_between(sessions.previous_session(clock.first_forward),
                                     sessions.previous_session(first.asof))
    bars = [day.isoformat() for day in days]
    rng = np.random.default_rng(21)
    equity = dict(zip(bars, 1e5 * np.cumprod(1.0 + rng.normal(0.0002, 0.004, len(bars)))))
    close = pd.DataFrame({"SGOV": 100.0 + rng.normal(0.0, 0.01, len(bars)),
                          "XYZ": 50.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, len(bars)))},
                         index=pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in days]))
    fills = [{"filled_bar": bars[0], "symbol": "SGOV", "qty": 200.0},
             {"filled_bar": bars[0], "symbol": "XYZ", "qty": -30.0}]
    paid = pd.DataFrame({"ex_date": pd.to_datetime([bars[20], bars[41]]), "pay_date": pd.to_datetime([bars[22], bars[43]]),
                         "amount": [0.33, 0.34]})
    paper = _paper_run_dir(world["base"] / "paper", positions={"SGOV": 200.0, "XYZ": -30.0}, bars=bars,
                           equity=equity, close=close, fills=fills, tranches=4, base=1, distributions=paid,
                           distributions_root=root)
    store_before = _snapshot(world["store"])
    root_before = _snapshot(root)
    paper_before = _snapshot(paper)

    def main(*args: str) -> int:
        return fwd.main([*args, "--workspace", str(root)], fetch=fetch, today=dt.date(2026, 10, 16))

    # Before the first scheduled as-of nothing runs; out of order nothing runs either.
    assert main("read", "--asof", str(sessions.previous_session(first.asof))) == 2
    assert main("run", "--asof", str(second.asof), "--dataset", str(world["store"])) == 2
    assert f"needs {first.name}" in capsys.readouterr().err
    assert not (root / "trials").exists()

    # The forward run: one started and one finished row in the forward ledger, none in the set's own.
    assert main("run", "--asof", str(first.asof), "--dataset", str(world["store"])) == 0
    rows = list(TrialLog(root / "trials" / f"{fwd.FORWARD_SET}.jsonl").verify())
    assert [(r["arm_id"], r["event"], r["kind"]) for r in rows] == [("T21", "started", "new"),
                                                                    ("T21", "finished", "new")]
    assert not (root / "trials" / f"{pb.SELECTION_SET}.jsonl").exists()
    out = root / fwd.READS_ROOT / str(first.asof)
    record = json.loads((out / "rows" / "T21" / "row.json").read_text())
    assert record["mechanics"]["n4"]["pass"] is True and record["mechanics"]["target_gross_within_ceiling"]
    baseline = pb.load_baseline(root)
    registered_spec = pb.row_specification(baseline, baseline.row("T21"))["specification"]
    assert record["specification"]["specification"] == registered_spec, "the registered row, unchanged"
    assert main("run", "--asof", str(first.asof), "--dataset", str(world["store"])) == 2, "a re-run is --retry"

    # The read: once, with every condition on both streams, and the latest French files recorded by hash.
    assert main("read", "--asof", str(first.asof), "--paper-run-dir", str(paper)) == 0
    summary = json.loads((out / "read.json").read_text())
    stream = summary["streams"]["forward"]
    assert set(stream["conditions"]) == {"a_interval_below_zero", "b_drawdown_above_limit", "c_fidelity_below_floor",
                                         "d_economics_below_hurdle", "d_two_consecutive_reads"}
    assert stream["n"] == first.threshold and stream["first"] == str(clock.first_forward)
    assert stream["sessions_before_every_tranche_scaled"] == 0
    assert stream["mechanics"]["n4_pass"] is True and stream["run_id"] == record["run_id"]
    assert stream["d_economics"]["figure"] == pytest.approx(stream["d_economics"]["figure_from_economics_of"])
    assert stream["conditions"]["d_two_consecutive_reads"] is False, "a first read has no previous read"
    assert stream["e_report_only"]["read"]["method"] == "empirical_bernstein"
    assert summary["eligibility"]["removed"] == stream["removes_eligibility"]
    momentum = summary["reference"]["files"]["momentum"]
    assert momentum["sha256"] == hashlib.sha256(world["payloads"][pb.FRENCH_URLS["ff_mom_daily"]]).hexdigest()
    assert momentum["vintage_line"] == VINTAGE_LINE and Path(momentum["path"]).is_file()
    assert summary["reference"]["rf_annual"] == pytest.approx(0.00015 * 252)
    on_paper = summary["streams"]["paper"]
    assert on_paper["n"] == first.threshold and set(on_paper["conditions"]) == set(stream["conditions"])
    assert on_paper["d_economics"]["cost_drag_sharpe"] == stream["d_economics"]["cost_drag_sharpe"]
    assert on_paper["d_economics"]["governing_hurdle_sharpe"] == stream["d_economics"]["governing_hurdle_sharpe"]
    corrections = on_paper["corrections"]
    assert corrections["etf"] == "SGOV" and corrections["distributions_total"] == pytest.approx(200 * (0.33 + 0.34))
    assert corrections["borrow_total"] > 0 and corrections["borrow_annual"] == fwd.PAPER_BORROW_ANNUAL
    written = pd.read_csv(out / "paper_stream.csv", index_col="date")
    assert len(written) == first.threshold
    # The paper stream's (e) is its corrected NAV return less the bill rate: the ETF is already in the NAV.
    rf_day = summary["reference"]["rf_annual"] / 252
    assert on_paper["e_report_only"]["read"]["mean"] == pytest.approx((written["net"] - rf_day).mean())
    assert summary["eligibility"]["by_stream"] == {"forward": stream["removes_eligibility"],
                                                   "paper": on_paper["removes_eligibility"]}
    assert main("read", "--asof", str(first.asof)) == 2, "a read is taken once"
    assert "already recorded" in capsys.readouterr().err

    # Nothing was written outside the declared places; the dataset and the paper run directory were only read.
    assert _snapshot(world["store"]) == store_before
    assert _snapshot(paper) == paper_before
    declared = [root / fwd.READS_ROOT, root / "trials", root / "state", root / "runs", root / "logs"]
    for path, digest in _snapshot(root).items():
        if root_before.get(path) == digest:
            continue
        candidate = Path(path)
        in_results_run = candidate.parent.parent == root / "results" and candidate.parent.name.startswith("run-")
        assert in_results_run or any(place in candidate.parents for place in declared), path
