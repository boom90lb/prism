"""Preflight doctor (prism.scripts.doctor): offline checks and exit semantics.

The network probes (Alpaca account read, Twelve Data quote) stay untested
here by design — each is a thin call into a client already tested offline
(tests/test_live_alpaca.py, tests/test_data_loader.py).
"""

from __future__ import annotations

import json

from prism.live.state import LoopState, StateStore
from prism.scripts.doctor import (
    check_env_credentials,
    check_kill_switch,
    check_loop_state,
    check_universe_file,
    main,
    run_checks,
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
# End-to-end offline run + exit semantics
# ---------------------------------------------------------------------------


def test_run_checks_offline_composes(tmp_path):
    universe = tmp_path / "u.txt"
    universe.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    results = _by_name(
        run_checks(
            run_dir=tmp_path / "run",
            universe_file=universe,
            data_dir=tmp_path / "data",
            env={"APCA_API_KEY_ID": "x", "APCA_API_SECRET_KEY": "y", "TWELVEDATA_API_KEY": "z"},
        )
    )
    assert {r.status for r in results.values()} == {"PASS"}
    # Offline run never includes the network probes.
    assert "alpaca-account" not in results


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
