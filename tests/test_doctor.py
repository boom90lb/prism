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

import functools
import json
from datetime import date

from prism.live.state import LoopState, StateStore
from prism.scripts import doctor
from prism.scripts.doctor import (
    CLEAN_SESSIONS_REQUIRED,
    UNIVERSE_STALE_DAYS,
    _repo_ops_file,
    check_account_book,
    check_alert_transport,
    check_artifact_sync,
    check_env_credentials,
    check_equity_ledger,
    check_holdings_priceable,
    check_kill_switch,
    check_loop_state,
    check_nightly_log,
    check_panel_record,
    check_regime_clock,
    check_shadow_record,
    check_universe_file,
    check_wrapper_provenance,
    main,
    run_checks,
    universe_as_of,
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
    assert "no 'as of' date" in result.detail  # a hand-written list has no header: legitimate, not stale


def _built_universe(path, as_of, n=150):
    """The header prism-build-universe writes (prism.scripts.build_sp500_universe), then symbols."""
    header = (
        f"# Current sp500 members as of {as_of} with local price data (fetchable\n"
        "# proxy for the live loop; NOT survivorship-free -- backtests use sp500_pit_*).\n"
        f"# Built from sp500_membership_{as_of}.parquet (intervals open at as-of)\n"
        f"# INTERSECT data/*_1d_*.parquet, minus quarantined symbols. {n} names.\n"
    )
    path.write_text(header + "\n".join(f"SYM{i}" for i in range(n)) + "\n", encoding="utf-8")


def test_universe_as_of_reads_the_header_only(tmp_path):
    path = tmp_path / "u.txt"
    _built_universe(path, "2026-09-25")
    assert universe_as_of(path) == date(2026, 9, 25)
    # A date past the header is a symbol line's business, not the file's age.
    path.write_text("AAA\n# as of 2026-01-01\nBBB\n", encoding="utf-8")
    assert universe_as_of(path) is None


def test_universe_file_fresh_build_passes_with_its_age(tmp_path):
    path = tmp_path / "u.txt"
    _built_universe(path, "2026-09-25")
    result = check_universe_file(path, today=date(2026, 9, 30))
    assert result.status == "PASS" and "as of 2026-09-25 (5 days ago)" in result.detail


def test_universe_file_stale_build_warns(tmp_path):
    # The 2026-07-17 build traded on 2026-09-25 holding shorts in two index leavers.
    path = tmp_path / "u.txt"
    _built_universe(path, "2026-07-17")
    result = check_universe_file(path, today=date(2026, 9, 25))
    assert result.status == "WARN"
    assert f"older than {UNIVERSE_STALE_DAYS} days" in result.detail and "150 symbols" in result.detail
    # The threshold is strict: exactly 45 days old still passes.
    assert check_universe_file(path, today=date(2026, 8, 31)).status == "PASS"
    assert check_universe_file(path, today=date(2026, 9, 1)).status == "WARN"


def test_universe_file_thin_and_stale_keeps_the_count_verdict(tmp_path):
    path = tmp_path / "u.txt"
    _built_universe(path, "2026-01-01", n=5)
    result = check_universe_file(path, today=date(2026, 9, 30))
    assert result.status == "WARN" and "decile" in result.detail and "as of 2026-01-01" in result.detail


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


def test_live_endpoint_is_refused_under_paper_mode():
    """§3.10: the endpoint check is a refusal under the wrong mode, not a warning.

    A doctor that reports "the shell points this loop at real money" as a WARN
    has already failed: the wrapper's exit code is what the alert reads, and a
    WARN does not change it.
    """
    env = {"APCA_API_BASE_URL": "https://api.alpaca.markets"}
    results = _by_name(check_env_credentials(env, mode="paper"))
    assert results["alpaca-endpoint"].status == "FAIL"
    assert "live endpoint under mode 'paper'" in results["alpaca-endpoint"].detail


def test_paper_endpoint_is_refused_under_live_mode():
    """And the reverse: --account live over the paper endpoint is refused too."""
    env = {"APCA_API_BASE_URL": "https://paper-api.alpaca.markets"}
    results = _by_name(check_env_credentials(env, mode="live"))
    assert results["alpaca-endpoint"].status == "FAIL"
    assert "paper endpoint under mode 'live'" in results["alpaca-endpoint"].detail


def test_endpoint_passes_on_its_own_mode():
    for mode, base in (("paper", "https://paper-api.alpaca.markets"), ("live", "https://api.alpaca.markets")):
        results = _by_name(check_env_credentials({"APCA_API_BASE_URL": base}, mode=mode))
        assert results["alpaca-endpoint"].status == "PASS"


# ---------------------------------------------------------------------------
# Alert transport — can anything leave the machine? (presence only)
# ---------------------------------------------------------------------------


def test_alert_transport_unset_warns():
    result = check_alert_transport({})
    assert result.status == "WARN" and "no alert can leave this machine" in result.detail
    assert check_alert_transport({"PRISM_HEARTBEAT_URL_SWEEP": "https://x/y"}).status == "WARN"


def test_alert_transport_present_passes_without_echoing_values():
    env = {"PRISM_ALERT_CMD": "/secret/path/alert.sh", "PRISM_HEARTBEAT_URL": "https://hc.example/SECRET-UUID"}
    result = check_alert_transport(env)
    assert result.status == "PASS"
    assert "PRISM_ALERT_CMD" in result.detail and "PRISM_HEARTBEAT_URL" in result.detail
    for value in env.values():
        assert value not in result.detail
    assert check_alert_transport({"PRISM_HEARTBEAT_URL": "https://hc.example/x"}).status == "PASS"
    assert check_alert_transport({"PRISM_ALERT_CMD": ""}).status == "WARN"  # empty is unset


# ---------------------------------------------------------------------------
# Artifact sync — the off-box backup's own report
# ---------------------------------------------------------------------------


def _sync_status(run_dir, **fields):
    run_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "at": "2026-09-30T01:31:00+00:00",
        "ok": True,
        "stage": "done",
        "detail": "committed and pushed abc1234; 4 file(s) compressed",
        "head": "abc1234",
        "pushed": True,
        "compressed": 4,
    }
    status.update(fields)
    (run_dir / "sync_status.json").write_text(json.dumps(status), encoding="utf-8")


def test_artifact_sync_absent_warns(tmp_path):
    result = check_artifact_sync(tmp_path, today=date(2026, 9, 30))
    assert result.status == "WARN" and "no backup has reported" in result.detail


def test_artifact_sync_failure_fails_naming_the_stage(tmp_path):
    # The 2026-09-25..30 condition, as the new script reports it.
    _sync_status(
        tmp_path,
        ok=False,
        stage="push",
        detail="push refused by the remote's file size limit (GH001): 4 unpushed local commit(s)",
        pushed=False,
    )
    result = check_artifact_sync(tmp_path, today=date(2026, 9, 30))
    assert result.status == "FAIL" and "stage 'push'" in result.detail and "GH001" in result.detail


def test_artifact_sync_fresh_success_passes_with_head(tmp_path):
    # Noon UTC keeps the local date the same in every zone the suite might run in.
    _sync_status(tmp_path, at="2026-09-29T12:00:00+00:00")
    result = check_artifact_sync(tmp_path, today=date(2026, 9, 30))
    assert result.status == "PASS" and "abc1234" in result.detail and "1 weekday(s) ago" in result.detail
    assert check_artifact_sync(tmp_path, today=date(2026, 9, 29)).status == "PASS"


def test_artifact_sync_stale_success_warns(tmp_path):
    _sync_status(tmp_path, at="2026-09-25T12:00:00+00:00")  # a Friday
    assert check_artifact_sync(tmp_path, today=date(2026, 9, 28)).status == "PASS"  # Monday: one weekday
    result = check_artifact_sync(tmp_path, today=date(2026, 9, 29))
    assert result.status == "WARN" and "stale" in result.detail


def test_artifact_sync_unreadable_status_fails(tmp_path):
    (tmp_path / "sync_status.json").write_text("{not json", encoding="utf-8")
    assert check_artifact_sync(tmp_path, today=date(2026, 9, 30)).status == "FAIL"
    _sync_status(tmp_path, at="yesterday-ish")
    assert check_artifact_sync(tmp_path, today=date(2026, 9, 30)).status == "FAIL"


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
    _nightly(
        tmp_path, [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={tmp_path}/stale_copy.sh commit=abc1234"]
    )
    result = check_wrapper_provenance(tmp_path)
    assert result.status == "FAIL" and "drifted copy" in result.detail


def test_wrapper_provenance_repo_wrapper_passes(tmp_path):
    _nightly(
        tmp_path,
        [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234"],
    )
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
    _nightly(
        tmp_path,
        [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234"],
    )
    _sweep(tmp_path, ["2026-07-31T06:50:00-0700 SWEEP: run-dir=runs/x", "2026-07-31T06:50:02-0700 EXIT 0"])
    result = check_wrapper_provenance(tmp_path)
    assert result.status == "FAIL" and "sweep.log" in result.detail


def test_wrapper_provenance_both_stamped_passes(tmp_path):
    _nightly(
        tmp_path,
        [f"2026-07-31T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234"],
    )
    _sweep(
        tmp_path,
        [f"2026-07-31T06:50:00-0700 SWEEP: run-dir=runs/x wrapper={_ops('paper_sweep_morning.sh')} commit=abc1234"],
    )
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


def _sessions(first, n):
    from prism.io import sessions

    out = [first]
    while len(out) < n:
        out.append(sessions.next_session(out[-1]))
    return out


def test_regime_clock_counts_the_trailing_streak_only(tmp_path):
    # A dirty session restarts the count; earlier clean sessions do not carry.
    days = _sessions(date(2026, 6, 1), 9)
    rows = [(d.isoformat(), i != 5) for i, d in enumerate(days)]
    _regime(tmp_path, rows)
    result = check_regime_clock(tmp_path, today=date(2026, 6, 12))
    assert result.status == "WARN" and f"3/{CLEAN_SESSIONS_REQUIRED}" in result.detail
    assert "is not clean" in result.detail


def test_regime_clock_satisfied_passes(tmp_path):
    days = _sessions(date(2026, 6, 1), CLEAN_SESSIONS_REQUIRED)
    _regime(tmp_path, [(d.isoformat(), True) for d in days])
    result = check_regime_clock(tmp_path, today=days[-1])
    assert result.status == "PASS" and "satisfied" in result.detail


def test_a_session_with_no_row_inside_the_run_restarts_it(tmp_path):
    # docs/regime_step.md §4: absence of the record is absence of the session.
    days = _sessions(date(2026, 6, 1), 8)
    _regime(tmp_path, [(d.isoformat(), True) for i, d in enumerate(days) if i != 4])
    result = check_regime_clock(tmp_path, today=days[-1])
    assert f"3/{CLEAN_SESSIONS_REQUIRED}" in result.detail and "no row for the session(s)" in result.detail


def test_unrecorded_sessions_since_the_last_row_zero_the_clock(tmp_path):
    # The 2026-08-14 -> 09-25 outage in miniature: 12 clean rows, then the loop went dark.
    days = _sessions(date(2026, 7, 30), 12)
    _regime(tmp_path, [(d.isoformat(), True) for d in days])
    assert days[-1] == date(2026, 8, 14)
    dark = check_regime_clock(tmp_path, today=date(2026, 9, 25))
    assert dark.status == "WARN" and dark.detail.startswith(f"0/{CLEAN_SESSIONS_REQUIRED}")
    assert "28 session(s) since 2026-08-14 have no regime row" in dark.detail and "was 12" in dark.detail
    # The evening of the last row and the next morning still stand at 12; a skipped day zeroes it.
    assert f"12/{CLEAN_SESSIONS_REQUIRED}" in check_regime_clock(tmp_path, today=date(2026, 8, 14)).detail
    assert f"12/{CLEAN_SESSIONS_REQUIRED}" in check_regime_clock(tmp_path, today=date(2026, 8, 17)).detail
    assert check_regime_clock(tmp_path, today=date(2026, 8, 18)).detail.startswith(f"0/{CLEAN_SESSIONS_REQUIRED}")
    # The first session after re-arming starts a new run at 1.
    _regime(tmp_path, [(d.isoformat(), True) for d in days] + [("2026-09-25", True)])
    assert f"1/{CLEAN_SESSIONS_REQUIRED}" in check_regime_clock(tmp_path, today=date(2026, 9, 25)).detail


def test_a_rerun_row_for_the_same_bar_replaces_the_first(tmp_path):
    days = _sessions(date(2026, 6, 1), 3)
    _regime(tmp_path, [(days[0].isoformat(), True), (days[1].isoformat(), False), (days[1].isoformat(), True),
                       (days[2].isoformat(), True)])
    assert f"3/{CLEAN_SESSIONS_REQUIRED}" in check_regime_clock(tmp_path, today=days[-1]).detail


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
    # 12 days before the injected `today`, 45+ days stale on the wall clock: the
    # universe row passes only if run_checks hands `today` through.
    _built_universe(universe, "2026-07-17")
    run_dir = tmp_path / "run"
    _equity(run_dir, ["2026-07-29"])
    _nightly(
        run_dir,
        [
            f"2026-07-29T18:30:01-0700 RUN: book=momentum wrapper={_ops('paper_loop_nightly.sh')} commit=abc1234",
            "2026-07-29T18:30:16-0700 EXIT 0",
        ],
    )
    _regime(run_dir, [(d.isoformat(), True) for d in _sessions(date(2026, 6, 30), CLEAN_SESSIONS_REQUIRED)])
    _sync_status(run_dir, at="2026-07-29T12:00:00+00:00")
    _panel_index(run_dir, ["2026-07-29"])
    _shadow(run_dir, ["2026-07-29"])
    results = _by_name(
        run_checks(
            run_dir=run_dir,
            universe_file=universe,
            data_dir=tmp_path / "data",
            env={
                "APCA_API_KEY_ID": "x",
                "APCA_API_SECRET_KEY": "y",
                "TWELVEDATA_API_KEY": "z",
                "PRISM_ALERT_CMD": "/x/alert.sh",
            },
            today=date(2026, 7, 29),
        )
    )
    assert {r.status for r in results.values()} == {"PASS"}
    # The injected date reached check_universe_file (a reversion to date.today() reads this build as stale).
    assert "as of 2026-07-17 (12 days ago)" in results["universe-file"].detail
    # Offline run never includes the network probes.
    assert "alpaca-account" not in results
    # …but it does include the health checks: a dark loop must be detectable
    # without credentials, since the alerting path cannot depend on the same
    # venue reachability whose absence it has to report.
    assert {
        "equity-ledger",
        "nightly-log",
        "wrapper-provenance",
        "regime-clock",
        "panel-record",
        "shadow-record",
        "artifact-sync",
        "alert-transport",
    } <= set(results)


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


def _pin_main_clock(monkeypatch) -> None:
    """Give ``main``'s checks a fixed date.

    ``main`` takes no date and calls ``run_checks`` without one, so every
    dated check reads ``date.today()``. The session-table row in particular
    FAILs once that is past the table's end, which would turn these exit-code
    tests red on that day with no change to the code.
    """
    monkeypatch.setattr(doctor, "run_checks", functools.partial(run_checks, today=date(2026, 7, 29)))


def test_main_exits_1_on_fail_and_0_on_pass(tmp_path, monkeypatch, capsys):
    _pin_main_clock(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "ACTIVE_RUN_DIR").write_text("runs/book\n", encoding="utf-8")
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
    assert "pass" in out and "fail" in out and "run-dir  runs/book" in out


def test_main_refuses_a_missing_or_empty_active_run_dir_instead_of_falling_back(tmp_path, monkeypatch, capsys):
    """Until 2026-09-30 a missing switch fell back to the retired runs/paper_loop_momentum and an empty one
    checked the current directory; the wrappers refuse both, and so does a hand-run doctor."""
    _pin_main_clock(monkeypatch)
    monkeypatch.chdir(tmp_path)
    universe = tmp_path / "u.txt"
    universe.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    assert main(["--universe-file", str(universe)]) == 1  # pointer absent
    out = capsys.readouterr().out
    assert "refused" in out and "ACTIVE_RUN_DIR" in out and "pass" not in out
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "ACTIVE_RUN_DIR").write_text("\n", encoding="utf-8")
    assert main(["--universe-file", str(universe)]) == 1  # pointer empty
    out = capsys.readouterr().out
    assert "refused" in out and "ACTIVE_RUN_DIR" in out
    assert not (tmp_path / "runs" / "paper_loop_momentum").exists()
    # An explicit --run-dir needs no switch.
    assert main(["--universe-file", str(universe), "--run-dir", "runs/book"]) == 0


# ---------------------------------------------------------------------------
# Panel record — did the last session record the bars it decided on?
# ---------------------------------------------------------------------------


def _panel_index(run_dir, bars):
    panels = run_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    (panels / "index.jsonl").write_text(
        "".join(json.dumps({"decision_bar": b, "kind": "delta", "sha256": "0" * 64}) + "\n" for b in bars),
        encoding="utf-8",
    )


def test_panel_record_absent_warns(tmp_path):
    _equity(tmp_path, ["2026-09-30"])
    result = check_panel_record(tmp_path, today=date(2026, 10, 1))
    assert result.status == "WARN" and "no session has recorded its panel since 2026-09-30" in result.detail


def test_panel_record_matching_the_ledger_passes(tmp_path):
    _equity(tmp_path, ["2026-09-30", "2026-10-01"])
    _panel_index(tmp_path, ["2026-09-30", "2026-10-01"])
    result = check_panel_record(tmp_path, today=date(2026, 10, 1))
    assert result.status == "PASS" and "2026-10-01" in result.detail


def test_panel_record_lagging_the_ledger_fails_and_counts_sessions(tmp_path):
    _equity(tmp_path, ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"])
    _panel_index(tmp_path, ["2026-09-30", "2026-10-01"])
    result = check_panel_record(tmp_path, today=date(2026, 10, 5))
    assert result.status == "FAIL" and "lags by 2 session(s)" in result.detail


def test_panel_record_ahead_of_the_ledger_warns(tmp_path):
    # The cycle recorded its inputs and died before the NAV mark.
    _equity(tmp_path, ["2026-09-30"])
    _panel_index(tmp_path, ["2026-09-30", "2026-10-01"])
    result = check_panel_record(tmp_path, today=date(2026, 10, 1))
    assert result.status == "WARN" and "did not complete" in result.detail


def test_panel_record_without_nav_rows_warns(tmp_path):
    _panel_index(tmp_path, ["2026-10-01"])
    assert check_panel_record(tmp_path, today=date(2026, 10, 1)).status == "WARN"


# ---------------------------------------------------------------------------
# Shadow record — did the last session write the row the tracking read needs?
# ---------------------------------------------------------------------------


def _shadow(run_dir, bars, paper=0.0010, shadow=0.0005):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "shadow.jsonl").write_text(
        "".join(
            json.dumps({"decision_bar": b, "paper_return": paper, "shadow_return": shadow, "transition": False}) + "\n"
            for b in bars
        ),
        encoding="utf-8",
    )


def _targets(run_dir, refresh_bars):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "targets.jsonl").write_text(
        "".join(json.dumps({"refresh_bar": b, "targets": {"AAA": 0.5}}) + "\n" for b in refresh_bars),
        encoding="utf-8",
    )


def test_shadow_record_absent_warns(tmp_path):
    _equity(tmp_path, ["2026-09-30"])
    result = check_shadow_record(tmp_path, today=date(2026, 10, 1))
    assert result.status == "WARN" and "no shadow row yet" in result.detail


def test_shadow_record_matching_the_ledger_passes_with_the_difference(tmp_path):
    _equity(tmp_path, ["2026-09-30", "2026-10-01"])
    _shadow(tmp_path, ["2026-09-30", "2026-10-01"])
    result = check_shadow_record(tmp_path, today=date(2026, 10, 1))
    assert result.status == "PASS" and "2026-10-01" in result.detail
    assert "paper minus shadow +0.000500" in result.detail


def test_shadow_record_lagging_a_session_that_owed_a_row_fails(tmp_path):
    # 2026-10-01 -> 10-02 are consecutive sessions and a refresh predates 10-01: the
    # cycle had everything it needed to shadow 10-02, so a missing row is a failed step.
    _equity(tmp_path, ["2026-09-30", "2026-10-01", "2026-10-02"])
    _shadow(tmp_path, ["2026-09-30", "2026-10-01"])
    _targets(tmp_path, ["2026-09-25"])
    result = check_shadow_record(tmp_path, today=date(2026, 10, 2))
    assert result.status == "FAIL" and "lags by 1 session(s)" in result.detail and "owed" in result.detail


def test_shadow_record_after_a_ledger_gap_warns_because_no_row_was_owed(tmp_path):
    # 10-02 and 10-05 are missing from the ledger: a gap is not a day, and the cycle
    # writes no shadow row across one (missed-sessions reports the gap itself).
    _equity(tmp_path, ["2026-09-30", "2026-10-01", "2026-10-06"])
    _shadow(tmp_path, ["2026-09-30", "2026-10-01"])
    _targets(tmp_path, ["2026-09-25"])
    result = check_shadow_record(tmp_path, today=date(2026, 10, 6))
    assert result.status == "WARN" and "not the preceding exchange session" in result.detail


def test_shadow_record_without_a_baseline_refresh_warns(tmp_path):
    # No refresh book decided before the previous bar: nothing to shadow with.
    _equity(tmp_path, ["2026-09-30", "2026-10-01", "2026-10-02"])
    _shadow(tmp_path, ["2026-09-30"])
    _targets(tmp_path, ["2026-10-01"])
    result = check_shadow_record(tmp_path, today=date(2026, 10, 2))
    assert result.status == "WARN" and "no refresh book" in result.detail
    (tmp_path / "targets.jsonl").unlink()
    assert check_shadow_record(tmp_path, today=date(2026, 10, 2)).status == "WARN"


def test_shadow_record_ahead_of_the_ledger_or_without_nav_rows_warns(tmp_path):
    _shadow(tmp_path, ["2026-10-01"])
    assert check_shadow_record(tmp_path, today=date(2026, 10, 1)).status == "WARN"
    _equity(tmp_path, ["2026-09-30"])
    result = check_shadow_record(tmp_path, today=date(2026, 10, 1))
    assert result.status == "WARN" and "ahead of the equity ledger" in result.detail


# ---------------------------------------------------------------------------
# Opt-in book features: margin, volatility scale, T-bill holding, tranches, seed
# ---------------------------------------------------------------------------


def _margin_read(**overrides):
    from prism.live.margin import AccountMargin

    values = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "long_market_value": 50_000.0,
        "short_market_value": 50_000.0,
        "initial_margin": 50_000.0,
        "maintenance_margin": 27_500.0,
        "regt_buying_power": 100_000.0,
        "buying_power": 100_000.0,
        "multiplier": 2.0,
    }
    values.update(overrides)
    return AccountMargin(**values)


def _margin_rows(run_dir, reads):
    """``reads``: (UTC datetime, AccountMargin) pairs, written as the margin report records them."""
    from prism.live.margin import MARGIN_LEDGER_FILENAME, append_margin_row, margin_row

    run_dir.mkdir(parents=True, exist_ok=True)
    for read_at, margin in reads:
        append_margin_row(
            run_dir / MARGIN_LEDGER_FILENAME, margin_row(margin, read_at=read_at, mode="paper", source="alpaca")
        )


def _evening(day):
    from datetime import datetime, timezone

    return datetime(day.year, day.month, day.day, 22, 0, tzinfo=timezone.utc)  # 18:00 ET


def _sizing(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sizing.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _single_sizing(bar, status, reason=None, scale=1.0, schedule="panel"):
    return {"decision_bar": bar, "status": status, "reason": reason, "scale": scale, "schedule": schedule}


def _holding_rows(run_dir, records):
    """``records``: (decision bar, held, target, decided) as the cycle writes them to ``cash_holding.jsonl``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "decision_bar": bar,
            "symbol": "SGOV",
            "long_weight": 0.5,
            "held_weight": held,
            "target_weight": target,
            "weight": decided,
            "step_bound": 0.199,
            "action": "hold" if decided == held else "rebalance",
            "wind_down": False,
            "buffer": 0.02,
            "band": 0.02,
        }
        for bar, held, target, decided in records
    ]
    (run_dir / "cash_holding.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_the_book_feature_checks_pass_on_a_run_directory_that_uses_none_of_them(tmp_path):
    from prism.scripts.doctor import (
        check_cash_holding_drift,
        check_deployment_profile,
        check_margin_cushion,
        check_tranche_state,
        check_vol_scale_fallback,
    )

    _equity(tmp_path, ["2026-10-15"])
    StateStore(tmp_path / "state.json").save(LoopState(positions={"AAA": 10.0}, cash=1e5))
    _targets(tmp_path, ["2026-10-01"])
    for check in (check_vol_scale_fallback, check_cash_holding_drift, check_tranche_state, check_deployment_profile):
        assert check(tmp_path).status == "PASS", check.__name__
    assert check_margin_cushion(tmp_path, today=date(2026, 10, 16)).status == "PASS"


def test_run_checks_includes_the_book_feature_checks(tmp_path):
    universe = tmp_path / "u.txt"
    universe.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    results = _by_name(
        run_checks(
            run_dir=tmp_path / "run",
            universe_file=universe,
            data_dir=tmp_path / "data",
            env={"APCA_API_KEY_ID": "x", "APCA_API_SECRET_KEY": "y"},
            today=date(2026, 10, 16),
        )
    )
    names = ("margin-cushion", "vol-scale-fallback", "cash-holding-drift", "tranche-state", "deployment-profile")
    assert {name: results[name].status for name in names} == {name: "PASS" for name in names}
    assert "alpaca-margin-cushion" not in results  # network only


# Margin cushion ---------------------------------------------------------------


def test_margin_cushion_absent_warns_only_beside_a_book_that_draws_on_margin(tmp_path):
    from prism.scripts.doctor import check_margin_cushion

    scaled = tmp_path / "scaled"
    _sizing(scaled, [_single_sizing("2026-10-01", "ok", scale=1.4)])
    result = check_margin_cushion(scaled, today=date(2026, 10, 16))
    assert result.status == "WARN" and "volatility-scaled" in result.detail and "--record" in result.detail
    holding = tmp_path / "holding"
    _holding_rows(holding, [("2026-10-01", 0.48, 0.48, 0.48)])
    result = check_margin_cushion(holding, today=date(2026, 10, 16))
    assert result.status == "WARN" and "T-bill ETF" in result.detail


def test_margin_cushion_reads_the_newest_recorded_row(tmp_path):
    from prism.scripts.doctor import check_margin_cushion

    _margin_rows(
        tmp_path,
        [
            (_evening(date(2026, 10, 14)), _margin_read(maintenance_margin=90_000.0)),
            (_evening(date(2026, 10, 15)), _margin_read()),
        ],
    )
    result = check_margin_cushion(tmp_path, today=date(2026, 10, 16))
    assert result.status == "PASS"
    assert "cushion 72.5%" in result.detail and "read 2026-10-15, 1 weekday(s) ago" in result.detail
    assert "gross 1.00x equity" in result.detail


def test_margin_cushion_below_the_floor_warns_and_says_nothing_acts_on_it(tmp_path):
    from prism.scripts.doctor import check_margin_cushion

    # Gross 2 at a house requirement of 40% on both sides: 0.8 of equity, a 20% cushion.
    thin = _margin_read(long_market_value=100_000.0, short_market_value=100_000.0, maintenance_margin=80_000.0)
    _margin_rows(tmp_path, [(_evening(date(2026, 10, 15)), thin)])
    result = check_margin_cushion(tmp_path, today=date(2026, 10, 16))
    assert (
        result.status == "WARN" and "cushion 20.0%" in result.detail and "below the 25% warning floor" in result.detail
    )
    assert "not implemented" in result.detail


def test_margin_cushion_stale_deficit_and_unreadable_reads_warn(tmp_path):
    from prism.scripts.doctor import check_margin_cushion

    stale = tmp_path / "stale"
    _margin_rows(stale, [(_evening(date(2026, 10, 9)), _margin_read())])
    result = check_margin_cushion(stale, today=date(2026, 10, 16))
    assert result.status == "WARN" and "stale" in result.detail
    deficit = tmp_path / "deficit"
    _margin_rows(deficit, [(_evening(date(2026, 10, 15)), _margin_read(equity=-1_000.0))])
    result = check_margin_cushion(deficit, today=date(2026, 10, 16))
    assert result.status == "WARN" and "deficit" in result.detail
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "margin.jsonl").write_text(json.dumps({"read_at": "x", "equity": 1.0}) + "\n", encoding="utf-8")
    result = check_margin_cushion(broken, today=date(2026, 10, 16))
    assert result.status == "WARN" and "unreadable" in result.detail


def test_the_live_margin_check_reads_the_venue_and_never_fails():
    from types import SimpleNamespace

    from prism.scripts.doctor import check_live_margin

    healthy = check_live_margin(SimpleNamespace(account_margin=lambda: _margin_read()))
    assert healthy.name == "alpaca-margin-cushion" and healthy.status == "PASS" and "venue, now" in healthy.detail
    thin = check_live_margin(SimpleNamespace(account_margin=lambda: _margin_read(maintenance_margin=80_000.0)))
    assert thin.status == "WARN"

    def broken():
        raise RuntimeError("GET /v2/account returned no maintenance_margin")

    failed = check_live_margin(SimpleNamespace(account_margin=broken))
    assert failed.status == "WARN" and "maintenance_margin" in failed.detail


def test_the_network_account_probe_reports_the_margin_cushion(monkeypatch):
    from types import SimpleNamespace

    import prism.live as live
    from prism.scripts.doctor import check_alpaca_account

    fake = SimpleNamespace(cash=lambda: 1e5, positions=lambda: {}, account_margin=lambda: _margin_read())
    monkeypatch.setattr(live.AlpacaBroker, "from_env", classmethod(lambda cls, **kwargs: fake))
    results = _by_name(check_alpaca_account())
    assert results["alpaca-account"].status == "PASS"
    assert (
        results["alpaca-margin-cushion"].status == "PASS"
        and "cushion 72.5%" in results["alpaca-margin-cushion"].detail
    )


# Volatility-scale fallbacks -----------------------------------------------------


def test_vol_scale_fallback_streak_of_two_warns_and_one_does_not(tmp_path):
    from prism.scripts.doctor import check_vol_scale_fallback

    one = tmp_path / "one"
    _sizing(
        one, [_single_sizing("2026-08-03", "ok", scale=1.4), _single_sizing("2026-09-01", "fallback", "zero_returns")]
    )
    result = check_vol_scale_fallback(one)
    assert result.status == "PASS" and "not yet a streak" in result.detail
    two = tmp_path / "two"
    _sizing(
        two,
        [
            _single_sizing("2026-08-03", "fallback", "insufficient_returns"),
            _single_sizing("2026-09-01", "ok", scale=1.4),
            _single_sizing("2026-09-30", "fallback", "zero_returns"),
            _single_sizing("2026-10-29", "fallback", "decision_bar_not_last_bar"),
        ],
    )
    result = check_vol_scale_fallback(two)
    assert result.status == "WARN"
    assert "2 consecutive refresh row(s) fell back, 2026-09-30 through 2026-10-29" in result.detail
    assert "decision_bar_not_last_bar" in result.detail and "zero_returns" in result.detail


def test_vol_scale_warmup_is_not_a_fallback(tmp_path):
    from prism.scripts.doctor import check_vol_scale_fallback

    _sizing(tmp_path, [_single_sizing(f"2026-0{m}-01", "warmup") for m in (7, 8, 9)])
    assert check_vol_scale_fallback(tmp_path).status == "PASS"


def test_vol_scale_fallback_counts_a_tranched_row_when_any_due_tranche_fell_back(tmp_path):
    from prism.scripts.doctor import check_vol_scale_fallback

    def row(bar, status):
        return {
            "decision_bar": bar,
            "schedule": "tranches",
            "account_scale": 1.2,
            "tranches": {
                "decile_neutral/t03": {"status": status, "reason": None if status == "ok" else "zero_returns"}
            },
        }

    _sizing(tmp_path, [row("2026-10-13", "ok"), row("2026-10-14", "fallback"), row("2026-10-15", "fallback")])
    result = check_vol_scale_fallback(tmp_path)
    assert result.status == "WARN" and "2 consecutive" in result.detail and "scale 1.200" in result.detail


# T-bill holding -------------------------------------------------------------------


def test_cash_holding_drift_warns_while_the_holding_is_built_and_passes_once_it_is(tmp_path):
    from prism.scripts.doctor import check_cash_holding_drift
    from tests.test_live_cash_holding import HOLDING, _ctx, _cycle, _panel

    close, volume = _panel()
    ctx = _ctx(tmp_path)
    _cycle(ctx, close, volume, 3, cash_holding=HOLDING)
    building = check_cash_holding_drift(tmp_path)
    assert building.status == "WARN" and "(bounded, band 0.0200)" in building.detail
    assert "within the per-order bound" in building.detail
    for t in range(4, 9):  # bounded, rebalance, then held inside the band
        _cycle(ctx, close, volume, t, cash_holding=HOLDING)
    built = check_cash_holding_drift(tmp_path)
    assert built.status == "PASS" and "BIL" in built.detail and "target 0.4800" in built.detail


def test_cash_holding_drift_warns_when_the_previous_sessions_order_did_not_fill(tmp_path):
    from prism.scripts.doctor import check_cash_holding_drift

    # 2026-10-14 decided 0.48; 2026-10-15 (the next session) marks 0.30: the order did not fill.
    _holding_rows(tmp_path, [("2026-10-14", 0.30, 0.48, 0.48), ("2026-10-15", 0.30, 0.48, 0.48)])
    result = check_cash_holding_drift(tmp_path)
    assert result.status == "WARN" and "did not fill as decided" in result.detail
    # Two records a month apart (the sessions in between did not run): no fill check.
    _holding_rows(tmp_path, [("2026-09-15", 0.30, 0.48, 0.48), ("2026-10-15", 0.47, 0.48, 0.47)])
    assert check_cash_holding_drift(tmp_path).status == "PASS"


def test_cash_holding_drift_reads_the_holding_between_a_monthly_books_refreshes(tmp_path):
    """The holding trades on sessions a monthly book does not refresh; the check reads those sessions too."""
    from dataclasses import replace

    from prism.scripts.doctor import check_cash_holding_drift
    from tests.test_live_cash_holding import CONFIG, HOLDING, _ctx, _cycle, _panel

    close, volume = _panel()
    ctx = _ctx(tmp_path)
    monthly = replace(CONFIG, decision_every=21)
    for t in range(3, 7):  # one refresh, then three sessions that only trade the holding
        _cycle(ctx, close, volume, t, cash_holding=HOLDING, config=monthly)
    assert len(_targets_rows(tmp_path)) == 1
    result = check_cash_holding_drift(tmp_path)
    assert result.status == "PASS" and f"at {close.index[6].date()}" in result.detail
    assert "decided 0.4800" in result.detail and "(hold, band 0.0200)" in result.detail


def _targets_rows(run_dir):
    path = run_dir / "targets.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# Tranche state ----------------------------------------------------------------------


def _tranched_run(run_dir, *, tranches, sessions_run, phase_offset=None):
    """A run directory the tranched cycle itself wrote (tests/test_live_tranches.py helpers)."""
    from tests.test_live_tranches import _ctx, _session, _walk

    close, volume = _walk(60)
    ctx = _ctx(run_dir)
    for i in range(30, 30 + sessions_run):
        _session(ctx, close, volume, i, tranches=tranches, phase_offset=phase_offset)
    return ctx


def test_tranche_state_passes_on_the_state_the_tranched_cycle_writes(tmp_path):
    from prism.scripts.doctor import check_tranche_state

    _tranched_run(tmp_path / "k21", tranches=21, sessions_run=22)
    result = check_tranche_state(tmp_path / "k21")
    assert result.status == "PASS", result.detail
    assert "21 tranche(s) of decile_neutral, 21 built, base offset 0, cycle 21 sessions" in result.detail
    _tranched_run(tmp_path / "k3", tranches=3, sessions_run=10, phase_offset=4)
    result = check_tranche_state(tmp_path / "k3")
    assert result.status == "PASS", result.detail
    assert "3 tranche(s)" in result.detail and "base offset 4, cycle 21 sessions" in result.detail


def test_tranche_state_warns_on_a_layout_the_cycle_could_not_have_written(tmp_path):
    from prism.scripts.doctor import check_tranche_state

    ctx = _tranched_run(tmp_path, tranches=3, sessions_run=10, phase_offset=4)
    state = ctx.store.load()
    state.components[2].phase_offset_bars = 19  # 18 is the third grid offset of base 4
    ctx.store.save(state)
    result = check_tranche_state(tmp_path)
    assert result.status == "WARN" and "grid offsets" in result.detail


def test_tranche_state_warns_when_the_targets_ledger_disagrees_with_the_state(tmp_path):
    from prism.scripts.doctor import check_tranche_state

    _tranched_run(tmp_path, tranches=21, sessions_run=3)
    with open(tmp_path / "targets.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"refresh_bar": "2099-01-02", "tranches": {"count": 3, "base_offset": 0}}) + "\n")
    result = check_tranche_state(tmp_path)
    assert result.status == "WARN" and "counts 3, the state 21" in result.detail


def test_inspect_tranche_state_names_each_defect():
    from prism.live.state import ComponentState
    from prism.live.tranches import inspect_tranche_state

    def tranche(j, offset, **overrides):
        values = {
            "component_id": f"decile_neutral/t{j:02d}",
            "last_output_bar": "2026-10-01",
            "expires_bar": "2026-10-30",
            "sub_targets": {"AAA": 0.25, "BBB": -0.25},
            "phase_offset_bars": offset,
            "holdings": {
                "AAA": {"shares": 10.0, "bar": "2026-10-01", "close": 100.0},
                "BBB": {"shares": -10.0, "bar": "2026-10-01", "close": 100.0},
            },
        }
        values.update(overrides)
        return ComponentState(**values)

    good = [tranche(0, 4), tranche(1, 11), tranche(2, 18)]
    report = inspect_tranche_state(LoopState(components=good, last_refresh_bar="2026-10-01"))
    assert report.problems == () and report.cycle == 21 and report.base_offset == 4
    assert abs(report.gross - 0.5) < 1e-12
    assert inspect_tranche_state(LoopState(components=[])) is None

    cases = {
        "persist no grid offset": [tranche(0, 4), tranche(1, None), tranche(2, 18)],
        "are not 00..02": [tranche(0, 4), tranche(1, 11), tranche(3, 18)],
        "non-finite sub-targets": [tranche(0, 4), tranche(1, 11, sub_targets={"AAA": float("nan")}), tranche(2, 18)],
        "non-positive last good scale": [
            tranche(0, 4, sizing={"scale": 1.0, "status": "ok", "last_good": -1.0}),
            tranche(1, 11),
            tranche(2, 18),
        ],
        "after the state's last refresh": [
            tranche(0, 4, last_output_bar="2026-10-02"),
            tranche(1, 11),
            tranche(2, 18),
        ],
        "before its output": [tranche(0, 4, expires_bar="2026-09-01"), tranche(1, 11), tranche(2, 18)],
        "persists no share holdings": [tranche(0, 4, holdings=None), tranche(1, 11), tranche(2, 18)],
        "sized after the state's last refresh": [
            tranche(0, 4, holdings={"AAA": {"shares": 1.0, "bar": "2026-10-02", "close": 100.0}}),
            tranche(1, 11),
            tranche(2, 18),
        ],
        "mixes tranche states": [*good, ComponentState(component_id="momentum")],
        "implemented for": [
            ComponentState(component_id=f"trend/t{j:02d}", phase_offset_bars=o) for j, o in enumerate((0, 7, 14))
        ],
    }
    for needle, components in cases.items():
        report = inspect_tranche_state(LoopState(components=components, last_refresh_bar="2026-10-01"))
        assert any(needle in problem for problem in report.problems), (needle, report.problems)


# Deployment profile -------------------------------------------------------------------


def test_deployment_profile_present_and_private_passes_without_showing_the_seed(tmp_path):
    from prism.live.deployment import DeploymentProfile, write_deployment
    from prism.scripts.doctor import check_deployment_profile

    profile = DeploymentProfile.new("2026-10-15")
    write_deployment(tmp_path, profile)
    result = check_deployment_profile(tmp_path)
    assert result.status == "PASS" and "created 2026-10-15, mode 600" in result.detail
    assert profile.seed_hex not in result.detail
    (tmp_path / "deployment.json").chmod(0o644)
    result = check_deployment_profile(tmp_path)
    assert result.status == "WARN" and "mode 644" in result.detail and profile.seed_hex not in result.detail


def test_deployment_profile_malformed_warns(tmp_path):
    from prism.scripts.doctor import check_deployment_profile

    (tmp_path / "deployment.json").write_text('{"schema": 1, "seed_hex": "short", "created_bar": "2026-10-15"}\n')
    (tmp_path / "deployment.json").chmod(0o600)
    result = check_deployment_profile(tmp_path)
    assert result.status == "WARN" and "malformed" in result.detail and "refuse" in result.detail


def test_deployment_profile_missing_beside_a_tranched_book_warns(tmp_path):
    from prism.scripts.doctor import check_deployment_profile

    _tranched_run(tmp_path / "epoch", tranches=21, sessions_run=2)
    result = check_deployment_profile(tmp_path / "epoch")
    assert result.status == "WARN" and "21 tranche state(s) at base offset 0" in result.detail
    assert "epoch grid" in result.detail
    _tranched_run(tmp_path / "seeded", tranches=3, sessions_run=10, phase_offset=4)
    result = check_deployment_profile(tmp_path / "seeded")
    assert result.status == "WARN" and "base offset 4" in result.detail and "will be refused" in result.detail


def test_deployment_profile_missing_beside_a_seeded_scaled_book_warns(tmp_path):
    from prism.scripts.doctor import check_deployment_profile

    _sizing(tmp_path, [_single_sizing("2026-10-01", "ok", scale=1.3, schedule="grid")])
    result = check_deployment_profile(tmp_path)
    assert result.status == "WARN" and "seeded grid" in result.detail
    _sizing(tmp_path, [_single_sizing("2026-10-01", "ok", scale=1.3, schedule="panel")])
    assert check_deployment_profile(tmp_path).status == "PASS"
