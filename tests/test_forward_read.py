"""Synthetic-data tests for the forward reader (research tier).

Nothing here reads a forward return: every clock's first read is scheduled for 2027. The tests pin the schedule
and the early refusal, the fold geometry and its hash, the derived registration and book, and each registered
statistic and verdict on synthetic series.
"""

import dataclasses
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts import forward_read as fr  # noqa: E402

FAST = {"block_bars": 42, "n_resamples": 200, "confidence": 0.95}


def _fast(clock):
    return dataclasses.replace(clock, bootstrap={**clock.bootstrap, **FAST})


def _index(first: dt.date, n: int) -> pd.DatetimeIndex:
    days = fr.session_dates(first, fr.shift_sessions(first, n - 1))
    return pd.DatetimeIndex(pd.to_datetime(days)).tz_localize(fr.TZ)


# --------------------------------------------------------------------- schedule and refusal


@pytest.mark.parametrize("name", sorted(fr.CLOCKS))
def test_every_read_is_refused_one_session_before_its_scheduled_asof(name):
    clock = fr.CLOCKS[name]
    reads = fr.scheduled_reads(clock)
    assert reads, name
    first = reads[0]
    assert first.threshold == 126
    with pytest.raises(fr.ReadRefused, match="Nothing was read"):
        fr.due_read(clock, fr.sessions.previous_session(first.asof))
    assert fr.due_read(clock, first.asof) == first
    # A read taken late is the scheduled one, at its scheduled as-of.
    assert fr.due_read(clock, first.asof + dt.timedelta(days=10)).asof == first.asof


def test_scheduled_dates_and_thresholds():
    got = {name: {r.name: (r.threshold, str(r.asof)) for r in fr.scheduled_reads(c)} for name, c in fr.CLOCKS.items()}
    assert got["forward_v1"] == {"interim": (126, "2027-03-29"), "promotion": (252, "2027-09-27")}
    assert got["earnings_daily_v1"] == got["forward_v1"]
    assert got["buffer_monthly_v1"] == {"interim": (126, "2027-03-30"), "promotion": (252, "2027-09-28")}
    assert got["buffer_daily_v1"] == got["buffer_monthly_v1"]
    # First kill read at 126, then every 63, as far as the bundled session table (ends 2027-12-31) can date.
    assert got["vol_momentum_v1"] == {"kill_1": (126, "2027-03-29"), "kill_2": (189, "2027-06-28"),
                                      "kill_3": (252, "2027-09-27"), "kill_4": (315, "2027-12-27")}


def test_status_today_reads_nothing_and_says_when():
    report = fr.status(dt.date(2026, 9, 25))
    assert all(entry["due"] is None and "first possible at as-of 2027" in entry["refusal"] for entry in report.values())
    assert report["forward_v1"]["elapsed_sessions_with_returns"] == 1
    assert report["buffer_daily_v1"]["elapsed_sessions_with_returns"] == 0


def test_cli_refuses_an_early_read_and_an_early_import(capsys):
    assert fr.main(["read", "--clock", "forward_v1", "--asof", "2026-09-25"]) == 2
    assert fr.main(["run", "--clock", "vol_momentum_v1", "--asof", "2027-03-26"]) == 2
    assert fr.main(["import", "--asof", "2026-09-25"]) == 2
    assert "REFUSED" in capsys.readouterr().err


# --------------------------------------------------------------------- geometry


def test_fold_table_is_the_certified_shape_and_its_hash_round_trips(tmp_path):
    clock = fr.CLOCKS["forward_v1"]
    (interim, promotion) = fr.scheduled_reads(clock)
    table = fr.fold_table_for(clock, promotion)
    assert len(table.folds) == 4
    for fold in table.folds:
        assert len(fr.session_dates(fold.formation_start.date(), fold.formation_end.date())) == fr.FORMATION
        assert len(fr.session_dates(fold.test_start.date(), fold.test_end.date())) == fr.TEST
    assert table.folds[0].test_start.date() == dt.date(2026, 9, 24)
    assert table.folds[-1].test_end.date() == fr.sessions.previous_session(promotion.asof)
    loaded, digest = fr.write_fold_table(table, tmp_path / "folds.toml")
    assert digest == table.content_hash == loaded.content_hash
    assert fr.fold_table_for(clock, interim).content_hash != digest


def test_warmup_is_three_windows_ending_the_session_before_the_first_forward_session():
    clock = fr.CLOCKS["vol_momentum_v1"]
    table = fr.fold_table_for(clock, fr.scheduled_reads(clock)[0], warmup=True)
    assert len(table.folds) == 3
    assert table.folds[-1].test_end.date() == dt.date(2026, 9, 23)
    assert sum(len(fr.session_dates(f.test_start.date(), f.test_end.date())) for f in table.folds) == 189


def test_the_warmup_lets_the_rule_scale_the_first_forward_session():
    from research.scripts.vol_momentum_replication import first_scored_index

    assert first_scored_index(fr.VOL_SOURCE) <= fr.VOL_WARMUP_FOLDS * fr.TEST
    assert first_scored_index(fr.VOL_SOURCE) > (fr.VOL_WARMUP_FOLDS - 1) * fr.TEST


# --------------------------------------------------------------------- derived registration and book


def test_derived_registration_changes_only_the_declared_keys_and_loads(tmp_path, monkeypatch):
    from prism.registration import load_registration
    from prism.workspace import resolve_workspace

    monkeypatch.setattr(fr, "READS_ROOT", tmp_path / "forward")
    clock = fr.CLOCKS["forward_v1"]
    read = fr.scheduled_reads(clock)[0]
    written = fr.plan(clock, read)
    path = fr.Path(written["registration"]["path"])
    source, derived = (fr.tomllib.loads(text) for text in (clock.registration.read_text(), path.read_text()))
    changed = {(section, key) for section in source for key in (source[section] if isinstance(source[section], dict)
               else [None]) if (source[section] if key is None else source[section][key])
               != (derived[section] if key is None else derived[section][key])}
    assert changed == {("id", None), ("created", None), ("selection_set", None), ("trial_budget", None),
                       ("arms", None), ("sample", "history_start"), ("sample", "first_decision"),
                       ("sample", "last_decision"), ("folds", "table"), ("frozen", "freeze_record")}
    kept = [arm for arm in source["arms"] if arm["id"] in {spec.arm for spec in clock.arms}]
    assert derived["arms"] == kept and len(kept) == 3
    registration = load_registration(path, workspace=resolve_workspace(None))
    assert registration.selection_set == "forward_v1" and registration.trial_budget == 3
    assert registration.fold_table.content_hash == written["fold_table"]["sha256"]
    assert registration.sample.first_decision == dt.date(2026, 9, 24)
    assert {"residual_momentum_only", "momentum_residual_momentum", "momentum_control"} <= set(registration.arm_ids)


def test_vol_plan_writes_a_warmup_and_a_forward_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "READS_ROOT", tmp_path / "forward")
    clock = fr.CLOCKS["vol_momentum_v1"]
    written = fr.plan(clock, fr.scheduled_reads(clock)[1])
    assert written["registration-warmup"]["selection_set"] == "vol_momentum_v1_warmup"
    assert written["registration"]["selection_set"] == "vol_momentum_v1_forward"
    assert written["fold_table-warmup"]["last_test"] == "2026-09-23"
    assert written["fold_table"]["n_folds"] == 3


def test_derived_book_changes_only_the_panel(tmp_path):
    source = fr.Path("examples/earnings/daily.toml").read_text()
    text = fr.derive_book(source, panel_path=tmp_path / "panel.parquet", panel_sha256="ab" * 32,
                          source_path=fr.Path("examples/earnings/daily.toml"))
    body = text.splitlines()[1:]
    diffs = [(a, b) for a, b in zip(source.splitlines(), body) if a != b]
    assert len(diffs) == 4 and all(a.split(" = ")[0] in {"panel", "panel_sha256"} for a, _ in diffs)
    with pytest.raises(fr.ReadRefused):
        fr.derive_book(source.replace(fr.REGISTERED_PANEL_SHA256, "00"), panel_path=tmp_path, panel_sha256="x",
                       source_path=fr.Path("x"))


def test_replace_lines_fails_loud_unless_the_count_matches():
    text = 'a = 1\nb = 2\nb = 3\n'
    assert fr.replace_lines(text, [(r"a = \d", "a = 9", 1)], where="t") == "a = 9\nb = 2\nb = 3\n"
    with pytest.raises(fr.ReadRefused):
        fr.replace_lines(text, [(r"b = \d", "b = 0", 1)], where="t")
    with pytest.raises(fr.ReadRefused):
        fr.replace_lines(text, [(r"c = \d", "c = 0", 1)], where="t")


# --------------------------------------------------------------------- statistics


def test_hurdle_is_the_amended_spec_formula():
    at_two_x = fr.tbill_hurdle(0.04, 0.033, 1.0, 0.10)
    assert at_two_x["sharpe_annual"] == pytest.approx(0.4) and at_two_x["idle_fraction"] == 0.0
    at_one_x = fr.tbill_hurdle(0.04, 0.033, 0.5, 0.10)
    assert at_one_x["sharpe_annual"] == pytest.approx((0.04 - 0.033 * 0.5) / 0.10)
    assert fr.tbill_hurdle(0.04, 0.033, 1.3, 0.10)["idle_fraction"] == 0.0
    assert at_one_x["sharpe_periodic"] == pytest.approx(at_one_x["sharpe_annual"] / math.sqrt(252))


def test_bond_equivalent_yield():
    assert fr.bond_equivalent_yield(pd.Series([5.0])).iloc[0] == pytest.approx(365 * 0.05 / (360 - 91 * 0.05))


def test_n6_reproduces_the_buffer_reads_recorded_ceilings():
    # docs/buffer_read_2026-09-25.md: premium 0.0090 (0.0034) at 5, N_eff 105.3 -> 0.65, 0.25 at lower-95;
    # momentum_buffered 0.030 (0.024) at 21, N_eff 44.1 -> 0.69, -0.21.
    premium = fr.n6_viability({"mean": 0.0090, "se": 0.0034}, 105.3, 5, 0.643, 0.183, 0.0)
    assert premium["ceiling"] == pytest.approx(0.65, abs=0.01)
    assert premium["ceiling_at_lower_95"] == pytest.approx(0.25, abs=0.01)
    assert premium["cost_drag"] == pytest.approx(0.46) and premium["viable"] is False
    momentum = fr.n6_viability({"mean": 0.030, "se": 0.024}, 44.1, 21, 0.686, 0.6, 0.0)
    assert momentum["ceiling"] == pytest.approx(0.69, abs=0.01)
    assert momentum["ceiling_at_lower_95"] == pytest.approx(-0.21, abs=0.01)
    assert fr.n6_viability({"mean": 0.05, "se": 0.005}, 100.0, 21, 0.8, 0.7, 0.3)["viable"] is True


def test_deflation_refuses_fewer_rows_than_registered():
    net = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 252))
    ok = fr.deflation(net, {f"r{i}": 0.02 + 0.001 * i for i in range(9)}, 9)
    assert 0.0 <= ok["dsr"] <= 1.0 and ok["n_rows"] == 9
    with pytest.raises(fr.ReadRefused):
        fr.deflation(net, {f"r{i}": 0.02 for i in range(8)}, 9)


def test_held_book_ic_is_one_when_weights_rank_the_next_returns():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2027-01-04", periods=80)
    names = [f"S{i}" for i in range(40)]
    growth = rng.normal(0.0, 0.01, (80, 40))
    opens = pd.DataFrame(100 * np.exp(np.cumsum(growth, axis=0)), index=days, columns=names)
    forward = opens.shift(-22) / opens.shift(-1) - 1.0
    decisions = forward.iloc[[0, 21, 42]].rank(axis=1) - 20.5
    got = fr.held_book_ic(decisions, opens, 21, 21)
    assert got["mean"] == pytest.approx(1.0) and got["n"] == 3 and got["nw_lags"] == 0
    assert fr.held_book_ic(decisions, opens, 5, 1)["nw_lags"] == 4


def test_max_drawdown_is_measured_from_the_forward_peak():
    assert fr.max_drawdown(pd.Series([0.10, -0.5, 0.2])) == pytest.approx(0.5)
    assert fr.max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(0.1)


# --------------------------------------------------------------------- verdicts


def _arm(net: np.ndarray, index: pd.DatetimeIndex, long: float = 0.5) -> fr.ArmInputs:
    weights = pd.DataFrame({"L": long, "S": -long}, index=index)
    return fr.ArmInputs(net=pd.Series(net, index=index), cost=pd.Series(0.0002, index=index), weights=weights,
                        decisions=weights.iloc[::21], n_eff=50.0)


def _context(verdict="inconclusive", clock=None):
    rows = {f"{src}:{arm}": 0.02 for src, arm in clock.deflation_rows if src != "forward"} if clock else {}
    return fr.ReadContext(rf_annual=0.04, cash_yield=0.033, monitor_verdict=verdict, trial_sharpes=rows)


def test_monitor_verdict_pins_its_method_and_records_the_reads_parameters(tmp_path, monkeypatch):
    """The conjunct's statistic is named by MONITOR_METHOD, not left to paper_monitor_read's default, and
    the packet carries what it was read with (live_monitor_design.md §4; N5)."""
    import prism.live.monitor as monitor

    seen = {}

    def fake_read(path, **kw):
        seen.update(path=path, **kw)
        return {"n": 16, "verdict": "inconclusive", "method": kw["method"], "bound": 0.1, "hurdle": 0.0,
                "alpha": 0.05, "n_gaps": 1, "mean": 0.0004, "ci_lower": -0.06, "ci_upper": 0.06,
                "skipped_pairs": [{"prev_bar": "2026-08-14", "bar": "2026-09-25"}]}

    monkeypatch.setattr(monitor, "paper_monitor_read", fake_read)
    pointer = tmp_path / "ACTIVE_RUN_DIR"
    pointer.write_text(str(tmp_path / "run") + "\n")
    monkeypatch.setattr(fr, "LIVE_RUN_DIR_POINTER", pointer)
    verdict, meta = fr.monitor_verdict()
    assert verdict == "inconclusive"
    assert seen["method"] == fr.MONITOR_METHOD == "empirical_bernstein"
    assert seen["path"] == tmp_path / "run" / "equity.jsonl"
    assert meta["run_dir"] == str(tmp_path / "run")
    assert {"method", "bound", "hurdle", "alpha", "n_gaps", "n", "verdict"} <= set(meta)
    assert meta["method"] == "empirical_bernstein" and meta["bound"] == 0.1 and meta["n_gaps"] == 1
    assert "skipped_pairs" not in meta  # the packet records parameters and the verdict, not the pair list


def test_forward_v1_interim_kills_on_the_paired_interval_only():
    clock = _fast(fr.CLOCKS["forward_v1"])
    read = fr.scheduled_reads(clock)[0]
    index = _index(dt.date(2026, 9, 24), read.threshold)
    rng = np.random.default_rng(11)
    base = rng.normal(0.001, 0.006, len(index))
    arms = {"momentum_control": _arm(base, index),
            "residual_momentum_only": _arm(base - 0.004 + rng.normal(0, 0.001, len(index)), index),
            "momentum_residual_momentum": _arm(base + rng.normal(0, 0.001, len(index)), index)}
    got = fr.decide(clock, read, arms, _context())
    assert got["killed"] == ["residual_momentum_only"]
    assert got["verdicts"]["momentum_control"] == {"role": "benchmark"}
    assert "criteria" not in got["verdicts"]["momentum_residual_momentum"]  # the interim cannot promote
    assert set(got["pairwise"]) == {"residual_momentum_only-momentum_residual_momentum",
                                    "residual_momentum_only-momentum_control", "momentum_residual_momentum-momentum_control"}
    stats = got["arms"]["momentum_control"]
    assert {"annualized_return", "psr_vs_zero", "turnover_annual", "max_drawdown", "n_eff"} <= set(stats)


def test_forward_v1_promotion_needs_every_criterion_and_nine_rows():
    clock = _fast(fr.CLOCKS["forward_v1"])
    read = fr.scheduled_reads(clock)[1]
    index = _index(dt.date(2026, 9, 24), read.threshold)
    rng = np.random.default_rng(12)
    base = rng.normal(0.0004, 0.006, len(index))
    arms = {"momentum_control": _arm(base, index),
            "residual_momentum_only": _arm(base + 0.0015 + rng.normal(0, 0.001, len(index)), index),
            "momentum_residual_momentum": _arm(base + 0.0008 + rng.normal(0, 0.001, len(index)), index)}
    got = fr.decide(clock, read, arms, _context(clock=clock))
    strong = got["verdicts"]["residual_momentum_only"]
    assert got["arms"]["residual_momentum_only"]["deflation"]["n_rows"] == 9
    assert set(strong["criteria"]) == {"net_edge", "dsr_above_half", "above_tbill_hurdle",
                                       "monitor_not_in_contradiction", "n6_viable"}
    assert strong["criteria"]["n6_viable"] is False and strong["promoted"] is False  # no IC without opens
    refuted = fr.decide(clock, read, arms, _context("refuted", clock=clock))
    assert refuted["verdicts"]["residual_momentum_only"]["criteria"]["monitor_not_in_contradiction"] is False


def test_own_interval_kill_for_the_daily_and_buffered_clocks():
    for name in ("earnings_daily_v1", "buffer_daily_v1", "buffer_monthly_v1"):
        clock = _fast(fr.CLOCKS[name])
        read = fr.scheduled_reads(clock)[0]
        index = _index(fr.first_forward_session(clock), read.threshold)
        rng = np.random.default_rng(13)
        loser, winner = (spec.arm for spec in clock.arms)
        arms = {loser: _arm(rng.normal(-0.003, 0.004, len(index)), index),
                winner: _arm(rng.normal(0.0005, 0.004, len(index)), index)}
        got = fr.decide(clock, read, arms, _context())
        assert got["killed"] == [loser], name


def test_promotion_with_opens_reports_ic_and_n6():
    clock = _fast(fr.CLOCKS["buffer_monthly_v1"])
    read = fr.scheduled_reads(clock)[1]
    index = _index(fr.first_forward_session(clock), read.threshold)
    naive = index.tz_localize(None)
    rng = np.random.default_rng(14)
    names = [f"S{i}" for i in range(40)]
    opens = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(index) + 30, 40)), axis=0)),
                         index=pd.DatetimeIndex(list(naive) + list(pd.bdate_range(naive[-1] + pd.Timedelta(days=1),
                                                                                    periods=30))), columns=names)
    arms = {}
    for spec in clock.arms:
        decisions = pd.DataFrame(rng.normal(0, 1, (len(naive[::21]), 40)), index=naive[::21], columns=names)
        arm = _arm(rng.normal(0.0006, 0.004, len(index)), index)
        arms[spec.arm] = dataclasses.replace(arm, decisions=decisions)
    context = _context(clock=clock)
    context.opens = opens
    got = fr.decide(clock, read, arms, context)
    for spec in clock.arms:
        stats = got["arms"][spec.arm]
        assert stats["held_book_ic"]["horizon"] == spec.horizon and stats["held_book_ic"]["n"] == 12
        assert stats["n6"]["horizon"] == spec.horizon and "viable" in stats["n6"]
        assert stats["deflation"]["n_rows"] == 8


def test_vol_kill_reads_the_scaled_forward_stream_and_its_three_conditions():
    clock = _fast(fr.CLOCKS["vol_momentum_v1"])
    read = fr.scheduled_reads(clock)[0]
    rng = np.random.default_rng(15)
    warm_index = pd.bdate_range("2026-01-02", periods=189)
    fwd_index = pd.bdate_range(warm_index[-1] + pd.Timedelta(days=1), periods=read.threshold)
    warm = pd.Series(rng.normal(0.0004, 0.008, 189), index=warm_index)
    healthy = rng.normal(0.0006, 0.008, read.threshold)
    forward = fr.ArmInputs(net=pd.Series(healthy, index=fwd_index), cost=pd.Series(0.0, index=fwd_index),
                           weights=pd.DataFrame(), decisions=pd.DataFrame(), n_eff=50.0)
    gross = pd.Series(1.0, index=warm_index.append(fwd_index))
    reference = pd.Series(healthy * 1.3 + rng.normal(0, 0.002, read.threshold), index=fwd_index)
    ok = fr.decide_vol(read, warm, forward, gross, reference, boot=clock.bootstrap)
    assert ok["killed"] is False and ok["reference_correlation"]["n_overlap"] == read.threshold
    assert 0.0 < ok["mean_scale"] <= fr.VOL_SOURCE["max_scale"]

    crash = healthy.copy()
    crash[40:70] = -0.02
    forward_crash = dataclasses.replace(forward, net=pd.Series(crash, index=fwd_index))
    bad = fr.decide_vol(read, warm, forward_crash, gross, reference, boot=clock.bootstrap)
    assert bad["conditions"]["b_drawdown_above_limit"] is True and bad["killed"] is True

    unrelated = pd.Series(rng.normal(0, 0.01, read.threshold), index=fwd_index)
    drift = fr.decide_vol(read, warm, forward, gross, unrelated, boot=clock.bootstrap)
    assert drift["conditions"]["c_fidelity_below_floor"] is True

    with pytest.raises(fr.ReadRefused):
        fr.decide_vol(read, warm, dataclasses.replace(forward, net=warm.iloc[-5:]), gross, reference)


def test_the_restated_vol_rule_matches_its_registration():
    fr.check_vol_registration()


@pytest.mark.parametrize("name", sorted(fr.CLOCKS))
def test_every_clocks_derived_registrations_load(name, tmp_path, monkeypatch):
    from prism.registration import load_registration
    from prism.workspace import resolve_workspace

    monkeypatch.setattr(fr, "READS_ROOT", tmp_path / "forward")
    clock = fr.CLOCKS[name]
    read = fr.scheduled_reads(clock)[-1]
    panel = (fr.Path(fr.REGISTERED_PANEL), fr.REGISTERED_PANEL_SHA256) if clock.book else None
    written = fr.plan(clock, read, panel)
    for suffix, _, selection_set in fr.jobs(clock):
        registration = load_registration(fr.Path(written[f"registration{suffix}"]["path"]),
                                         workspace=resolve_workspace(None))
        assert registration.selection_set == selection_set
        assert set(registration.arm_ids) == {spec.arm for spec in clock.arms}
        assert registration.fold_table.content_hash == written[f"fold_table{suffix}"]["sha256"]
        if clock.book:
            assert registration.benchmark.control.name == clock.book.name
