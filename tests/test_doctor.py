"""Preflight + health doctor (prism.scripts.doctor): checks and exit semantics.

The network *transport* stays untested here by design — each probe is a thin
call into a client already tested offline (tests/test_live_alpaca.py,
tests/test_data_loader.py). The *verdicts* those probes compute do not:
``check_account_book`` and ``check_holdings_priceable`` are pure functions of a
position book, so the 2026-07-23 condition they exist to catch — a run
directory with no record of a live account, reported as "8 pass, 0 fail" — is
pinned here without a venue.
"""

from __future__ import annotations

import json
from datetime import date

from prism.live.state import LoopState, StateStore
from prism.scripts.doctor import (
    CLEAN_SESSIONS_REQUIRED,
    check_account_book,
    check_env_credentials,
    check_equity_ledger,
    check_holdings_priceable,
    check_kill_switch,
    check_loop_state,
    _repo_ops_file,
    check_nightly_log,
    check_regime_clock,
    check_universe_file,
    check_wrapper_provenance,
    main,
    run_checks,
    weekdays_since,
)


def _by_name(results):
    return {r.name: r for r in results}


# ---------------------------------------------------------------------------
# Universe file
# ---------------------------------------------------------------------------


def test_universe_file_missing_fails(tmp_path):
    assert check_universe_file(tmp_path / "nope.txt").status == "FAIL"


def test_universe_file_empty_fails(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("# only a comment\n\n", encoding="utf-8")
    assert check_universe_file(path).status == "FAIL"


def test_universe_file_thin_warns(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("\n".join(f"SYM{i}" for i in range(5)), encoding="utf-8")
    result = check_universe_file(path)
    assert result.status == "WARN" and "decile" in result.detail


def test_universe_file_full_passes(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    result = check_universe_file(path)
    assert result.status == "PASS" and "150 symbols" in result.detail


# ---------------------------------------------------------------------------
# Credentials (presence only — values never printed, docs/security.md)
# ---------------------------------------------------------------------------


def test_missing_alpaca_keys_fail_and_are_not_echoed():
    results = _by_name(check_env_credentials({}))
    assert results["alpaca-credentials"].status == "FAIL"
    assert results["twelvedata-key"].status == "WARN"
    assert results["alpaca-endpoint"].status == "PASS"


def test_present_keys_pass_without_leaking_values():
    env = {
        "APCA_API_KEY_ID": "PKSECRETID",
        "APCA_API_SECRET_KEY": "sk-SECRET",
        "TWELVEDATA_API_KEY": "td-SECRET",
    }
    results = check_env_credentials(env)
    assert all(r.status == "PASS" for r in results)
    blob = " ".join(r.detail for r in results)
    for value in env.values():
        assert value not in blob  # presence is reported, values never are


def test_live_endpoint_warns():
    env = {"APCA_API_BASE_URL": "https://api.alpaca.markets"}
    results = _by_name(check_env_credentials(env))
    assert results["alpaca-endpoint"].status == "WARN"
    assert "real money" in results["alpaca-endpoint"].detail


# ---------------------------------------------------------------------------
# Loop state + kill switch
# ---------------------------------------------------------------------------


def test_loop_state_absent_is_fresh(tmp_path):
    assert check_loop_state(tmp_path).status == "PASS"


def test_loop_state_corrupt_fails(tmp_path):
    (tmp_path / "state.json").write_text("{not json", encoding="utf-8")
    result = check_loop_state(tmp_path)
    assert result.status == "FAIL" and "corrupt" in result.detail


def test_loop_state_valid_reports_book(tmp_path):
    StateStore(tmp_path / "state.json").save(
        LoopState(positions={"AAA": 10.0}, cash=5_000.0, last_refresh_bar="2026-07-01")
    )
    result = check_loop_state(tmp_path)
    assert result.status == "PASS"
    assert "1 positions" in result.detail and "2026-07-01" in result.detail


def test_kill_switch_present_warns(tmp_path):
    (tmp_path / "KILL_SWITCH").touch()
    result = check_kill_switch(tmp_path)
    assert result.status == "WARN" and "halted" in result.detail
    (tmp_path / "KILL_SWITCH").unlink()
    assert check_kill_switch(tmp_path).status == "PASS"


# ---------------------------------------------------------------------------
# Session age (holiday-blind by design — see weekdays_since)
# ---------------------------------------------------------------------------


def test_weekdays_since_counts_only_weekdays():
    assert weekdays_since(date(2026, 7, 29), date(2026, 7, 29)) == 0  # same day
    assert weekdays_since(date(2026, 7, 29), date(2026, 7, 30)) == 1
    # Fri -> Mon is one missed weekday, not three: a normal overnight gap.
    assert weekdays_since(date(2026, 7, 24), date(2026, 7, 27)) == 1
    # The outage window: last mark Thu 07-23, checked Wed 07-29.
    assert weekdays_since(date(2026, 7, 23), date(2026, 7, 29)) == 4
    assert weekdays_since(date(2026, 7, 30), date(2026, 7, 29)) == 0  # future mark


# ---------------------------------------------------------------------------
# Ledger freshness — the liveness signal a green scheduler cannot fake
# ---------------------------------------------------------------------------


def _equity(run_dir, bars):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "equity.jsonl").write_text(
        "".join(json.dumps({"decision_bar": b, "equity": 1e5, "cash": 1e5}) + "\n" for b in bars),
        encoding="utf-8",
    )


def test_equity_ledger_absent_warns_not_fails(tmp_path):
    # A genuinely fresh run directory is legitimate; it turns fatal only beside a
    # live venue book, which check_account_book decides.
    result = check_equity_ledger(tmp_path)
    assert result.status == "WARN" and "no NAV rows" in result.detail


def test_equity_ledger_fresh_passes(tmp_path):
    _equity(tmp_path, ["2026-07-28", "2026-07-29"])
    result = check_equity_ledger(tmp_path, today=date(2026, 7, 30))
    assert result.status == "PASS" and "2026-07-29" in result.detail


def test_equity_ledger_one_missed_session_warns(tmp_path):
    _equity(tmp_path, ["2026-07-27"])
    assert check_equity_ledger(tmp_path, today=date(2026, 7, 29)).status == "WARN"


def test_equity_ledger_dark_loop_fails(tmp_path):
    # The specimen: last completed cycle 2026-07-10, checked 2026-07-29.
    _equity(tmp_path, ["2026-07-08", "2026-07-10"])
    result = check_equity_ledger(tmp_path, today=date(2026, 7, 29))
    assert result.status == "FAIL" and "DARK" in result.detail


def test_equity_ledger_unparseable_bar_fails(tmp_path):
    _equity(tmp_path, ["not-a-date"])
    assert check_equity_ledger(tmp_path, today=date(2026, 7, 29)).status == "FAIL"


# ---------------------------------------------------------------------------
# Nightly log — nonzero verdicts, and stale successes
# ---------------------------------------------------------------------------


def _nightly(run_dir, lines):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "nightly.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_nightly_log_absent_warns(tmp_path):
    assert check_nightly_log(tmp_path).status == "WARN"


def test_nightly_log_nonzero_exit_fails_and_counts_the_streak(tmp_path):
    _nightly(
        tmp_path,
        [
            "2026-07-22T18:30:00-0700 RUN: book=momentum",
            "2026-07-22T18:30:16-0700 EXIT 0",
            "2026-07-23T18:30:00-0700 RUN: book=momentum",
            "2026-07-23T18:30:16-0700 EXIT 1",
            "2026-07-24T18:30:00-0700 RUN: book=momentum",
            "2026-07-24T18:30:16-0700 EXIT 1",
        ],
    )
    result = check_nightly_log(tmp_path, today=date(2026, 7, 24))
    assert result.status == "FAIL"
    assert "EXIT 1" in result.detail and "2 consecutive failed" in result.detail


def test_nightly_log_halt_exit_2_also_fails(tmp_path):
    # A halted book exits 2 — deliberate, and still a state that demands eyes.
    _nightly(tmp_path, ["2026-07-29T18:30:16-0700 EXIT 2"])
    assert check_nightly_log(tmp_path, today=date(2026, 7, 29)).status == "FAIL"


def test_nightly_log_stale_success_fails(tmp_path):
    # The failure mode that hid behind a green morning sweep: the last thing that
    # ran succeeded, and nothing has run since.
    _nightly(tmp_path, ["2026-07-10T18:30:16-0700 EXIT 0"])
    result = check_nightly_log(tmp_path, today=date(2026, 7, 29))
    assert result.status == "FAIL" and "STALE success" in result.detail


def test_nightly_log_recent_success_passes(tmp_path):
    _nightly(tmp_path, ["2026-07-28T18:30:16-0700 EXIT 0", "2026-07-29T18:30:16-0700 EXIT 0"])
    assert check_nightly_log(tmp_path, today=date(2026, 7, 29)).status == "PASS"


def test_nightly_log_skip_warns(tmp_path):
    _nightly(tmp_path, ["2026-07-29T18:30:00-0700 SKIP: APCA credentials not present"])
    result = check_nightly_log(tmp_path, today=date(2026, 7, 29))
    assert result.status == "WARN" and "SKIP" in result.detail


def test_nightly_log_without_verdicts_warns(tmp_path):
    _nightly(tmp_path, ["2026-07-29T18:30:00-0700 RUN: book=momentum"])
    assert check_nightly_log(tmp_path, today=date(2026, 7, 29)).status == "WARN"


# ---------------------------------------------------------------------------
# Wrapper provenance — what runs must be what was reviewed
# ---------------------------------------------------------------------------


def _sweep(run_dir, lines):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sweep.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ops(name):
    path = _repo_ops_file(name)
    assert path is not None, f"repo ops/{name} must exist for these tests"
    return path


def test_wrapper_provenance_no_sessions_warns(tmp_path):
    assert check_wrapper_provenance(tmp_path).status == "WARN"


def test_wrapper_provenance_unstamped_run_fails(tmp_path):
    # The 2026-07-31 condition: the committed alert layer never executed because
    # a stale pre-provenance ~/bin copy was scheduled — and nothing said so.
    _nightly(tmp_path, ["2026-07-30T18:30:01-0700 RUN: book=momentum regime=on", "2026-07-30T18:30:48-0700 EXIT 0"])
    result = check_wrapper_provenance(tmp_path)
    assert result.status == "FAIL" and "unstamped" in result.detail


def test_wrapper_provenance_foreign_path_fails(tmp_path):
    _nightly(tmp_path, [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={tmp_path}/stale_copy.sh commit=abc1234"])
    result = check_wrapper_provenance(tmp_path)
    assert result.status == "FAIL" and "drifted copy" in result.detail


def test_wrapper_provenance_repo_wrapper_passes(tmp_path):
    _nightly(tmp_path, [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234"])
    result = check_wrapper_provenance(tmp_path)
    assert result.status == "PASS" and "paper_loop_nightly.sh" in result.detail


def test_wrapper_provenance_only_latest_entry_counts(tmp_path):
    # Pre-provenance history is history; the invariant binds the *last* session.
    _nightly(
        tmp_path,
        [
            "2026-07-30T18:30:01-0700 RUN: book=momentum regime=on",
            "2026-07-30T18:30:48-0700 EXIT 0",
            f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234",
            "2026-07-31T18:30:48-0700 EXIT 0",
        ],
    )
    assert check_wrapper_provenance(tmp_path).status == "PASS"


def test_wrapper_provenance_stale_sweep_fails_even_with_good_nightly(tmp_path):
    _nightly(tmp_path, [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234"])
    _sweep(tmp_path, ["2026-07-31T06:50:00-0700 SWEEP: run-dir=runs/x", "2026-07-31T06:50:02-0700 EXIT 0"])
    result = check_wrapper_provenance(tmp_path)
    assert result.status == "FAIL" and "sweep.log" in result.detail


def test_wrapper_provenance_both_stamped_passes(tmp_path):
    _nightly(tmp_path, [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234"])
    _sweep(tmp_path, [f"2026-07-31T06:50:00-0700 SWEEP: run-dir=runs/x wrapper={_ops('paper_sweep_morning.sh')} commit=abc1234"])
    result = check_wrapper_provenance(tmp_path)
    assert result.status == "PASS" and "sweep" in result.detail


# ---------------------------------------------------------------------------
# Regime clock (handoff §8 precondition (b), docs/regime_step.md §4)
# ---------------------------------------------------------------------------


def _regime(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "regime.jsonl").write_text(
        "".join(json.dumps({"decision_bar": bar, "clean": clean}) + "\n" for bar, clean in rows),
        encoding="utf-8",
    )


def test_regime_clock_absent_warns_at_zero(tmp_path):
    result = check_regime_clock(tmp_path)
    assert result.status == "WARN" and f"0/{CLEAN_SESSIONS_REQUIRED}" in result.detail


def test_regime_clock_counts_the_trailing_streak_only(tmp_path):
    # A dirty session restarts the count; earlier clean sessions do not carry.
    rows = [(f"2026-06-{d:02d}", True) for d in range(1, 6)]
    rows += [("2026-06-06", False)] + [(f"2026-06-{d:02d}", True) for d in range(7, 10)]
    _regime(tmp_path, rows)
    result = check_regime_clock(tmp_path)
    assert result.status == "WARN" and f"3/{CLEAN_SESSIONS_REQUIRED}" in result.detail


def test_regime_clock_satisfied_passes(tmp_path):
    _regime(tmp_path, [(f"2026-06-{d:02d}", True) for d in range(1, 1 + CLEAN_SESSIONS_REQUIRED)])
    result = check_regime_clock(tmp_path)
    assert result.status == "PASS" and "satisfied" in result.detail


# ---------------------------------------------------------------------------
# Account reconciliation — the check whose absence reported "8 pass, 0 fail"
# ---------------------------------------------------------------------------


def test_unattributed_venue_book_fails(tmp_path):
    # The 2026-07-23 condition, exactly: a fresh run directory over a live book.
    result = check_account_book({"AAA": 10.0, "POOL": -5.0}, ["AAA", "BBB"], None, run_dir=tmp_path)
    assert result.status == "FAIL"
    assert "no record of the account's book" in result.detail
    assert "POOL" in result.detail  # the off-universe leaver is named


def test_flat_account_passes(tmp_path):
    assert check_account_book({}, ["AAA"], None, run_dir=tmp_path).status == "PASS"


def test_zero_share_rows_are_not_a_book(tmp_path):
    assert check_account_book({"AAA": 0.0}, ["AAA"], None, run_dir=tmp_path).status == "PASS"


def test_reconciled_book_passes_and_names_off_universe_holdings(tmp_path):
    state = LoopState(positions={"AAA": 10.0, "POOL": -5.0})
    result = check_account_book({"AAA": 10.0, "POOL": -5.0}, ["AAA"], state, run_dir=tmp_path)
    assert result.status == "PASS" and "POOL" in result.detail


def test_diverged_book_warns_naming_both_sides(tmp_path):
    # 28 persisted against 98 at the venue was the real drift; the shape is the
    # same at two names, and adopting broker truth is the loop's job, not a stop.
    result = check_account_book({"AAA": 10.0}, ["AAA", "BBB"], LoopState(positions={"BBB": 4.0}), run_dir=tmp_path)
    assert result.status == "WARN"
    assert "['AAA']" in result.detail and "['BBB']" in result.detail


def test_holdings_priceable_verdicts():
    assert check_holdings_priceable({}, {}).status == "PASS"
    assert check_holdings_priceable({"AAA": 1.0}, {"AAA": 101.5}).status == "PASS"
    # The mark step's three refusals: absent, None, non-positive.
    missing = check_holdings_priceable({"AAA": 1.0, "POOL": -5.0}, {"AAA": 101.5})
    assert missing.status == "FAIL" and "POOL" in missing.detail
    assert check_holdings_priceable({"POOL": -5.0}, {"POOL": None}).status == "FAIL"
    assert check_holdings_priceable({"POOL": -5.0}, {"POOL": 0.0}).status == "FAIL"


# ---------------------------------------------------------------------------
# End-to-end offline run + exit semantics
# ---------------------------------------------------------------------------


def test_run_checks_offline_composes(tmp_path):
    universe = tmp_path / "u.txt"
    universe.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    run_dir = tmp_path / "run"
    _equity(run_dir, ["2026-07-29"])
    _nightly(
        run_dir,
        [
            f"2026-07-29T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234",
            "2026-07-29T18:30:16-0700 EXIT 0",
        ],
    )
    _regime(run_dir, [(f"2026-07-{d:02d}", True) for d in range(1, 1 + CLEAN_SESSIONS_REQUIRED)])
    results = _by_name(
        run_checks(
            run_dir=run_dir,
            universe_file=universe,
            data_dir=tmp_path / "data",
            env={"APCA_API_KEY_ID": "x", "APCA_API_SECRET_KEY": "y", "TWELVEDATA_API_KEY": "z"},
            today=date(2026, 7, 29),
        )
    )
    assert {r.status for r in results.values()} == {"PASS"}
    # Offline run never includes the network probes.
    assert "alpaca-account" not in results
    # …but it does include the health checks: a dark loop must be detectable
    # without credentials, since the alerting path cannot depend on the same
    # venue reachability whose absence it has to report.
    assert {"equity-ledger", "nightly-log", "wrapper-provenance", "regime-clock"} <= set(results)


def test_run_checks_offline_fails_on_a_dark_loop(tmp_path):
    universe = tmp_path / "u.txt"
    universe.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    run_dir = tmp_path / "run"
    _equity(run_dir, ["2026-07-10"])
    _nightly(run_dir, ["2026-07-29T18:30:16-0700 EXIT 1"])
    results = _by_name(
        run_checks(
            run_dir=run_dir,
            universe_file=universe,
            data_dir=tmp_path / "data",
            env={"APCA_API_KEY_ID": "x", "APCA_API_SECRET_KEY": "y", "TWELVEDATA_API_KEY": "z"},
            today=date(2026, 7, 29),
        )
    )
    assert results["equity-ledger"].status == "FAIL"
    assert results["nightly-log"].status == "FAIL"


def test_main_exits_1_on_fail_and_0_on_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    universe = tmp_path / "u.txt"
    universe.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    # Missing Alpaca keys -> alpaca-credentials FAILs -> exit 1.
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert main(["--universe-file", str(universe)]) == 1
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    assert main(["--universe-file", str(universe)]) == 0
    out = capsys.readouterr().out
    assert "pass" in out and "fail" in out
