"""The stage 0 paper-state assessment (``prism.scripts.paper_state_assessment``).

The script exists because the paper loop went dark and somebody has to write
down what the run directory records. Its value is entirely in what it refuses
to do, so that is what these tests attack:

* it must never invent a session. The multi-session gap test pins the exact
  missed dates against the session table and asserts the ledgers grew no rows.
* it must never touch the run directory. One test byte-compares every file and
  its mtime across a run.
* it must never guess a schema, and the schema set it accepts must not drift
  away from the loader's (``prism/live/state.py``); one test pins the two.
* it must never need a credential. One test reads the module's own top-level
  imports and fails if the broker chain is among them; a second runs the CLI in
  a fresh interpreter holding sentinel Alpaca keys and fails if any
  ``prism.live`` module loads or either sentinel reaches the report.
* the regime streak it reports must be the doctor's streak, not a lookalike;
  one test computes both over the same ledger and compares.

Every fixture run directory is built in ``tmp_path``. No test reads the live
run directory.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest

from prism.io import sessions
from prism.scripts import paper_state_assessment as psa

ASOF = "2026-09-08"

# The fifteen sessions between the last completed one (2026-08-14) and the
# assessment date, written out here rather than derived, so a bug in the
# session table cannot agree with a bug in the script.
MISSED_2026 = (
    "2026-08-17",
    "2026-08-18",
    "2026-08-19",
    "2026-08-20",
    "2026-08-21",
    "2026-08-24",
    "2026-08-25",
    "2026-08-26",
    "2026-08-27",
    "2026-08-28",
    "2026-08-31",
    "2026-09-01",
    "2026-09-02",
    "2026-09-03",
    "2026-09-04",
)


def _state(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cash": 937669.47,
        "positions": {"AAA": 10.0, "BBB": -5.0, "CCC": 7.0},
        "pending_orders": [],
        "pending_decision_bar": "2026-08-14",
        "last_settled_bar": "2026-08-11",
        "last_refresh_bar": "2026-08-11",
        "schema_version": 2,
    }
    payload.update(overrides)
    return payload


def _nightly_log(*, run_date: str, commit: str = "abc1234", exit_code: int = 0) -> str:
    return "\n".join(
        [
            f"{run_date}T18:30:00-0700 RUN: book=momentum universe=data/universe/u.txt regime=on "
            f"wrapper=/repo/ops/paper_loop_nightly.sh commit={commit}",
            f"{run_date} 18:30:19,988 INFO __main__ panels: 753 bars x 3 symbols",
            f"{run_date}T18:30:27-0700 EXIT {exit_code}",
            f"{run_date}T18:30:27-0700 DOCTOR:",
            "run-dir  runs/fixture",
            "PASS  loop-state                 3 positions",
            "WARN  regime-clock               2/21 consecutive clean sessions",
            "",
            "15 pass, 1 warn, 0 fail",
            f"{run_date}T18:30:30-0700 DOCTOR EXIT 0",
            "",
        ]
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build_run_dir(
    root: Path,
    *,
    bars: Sequence[str] = ("2026-08-12", "2026-08-13", "2026-08-14"),
    clean: Sequence[bool] | None = None,
    state: dict[str, Any] | None = None,
    nightly: str | None = None,
    empty_ledgers: bool = False,
) -> Path:
    """A run directory shaped like the live one: state, six ledgers, a nightly log."""
    run_dir = root / "runs" / "fixture"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps(state if state is not None else _state(), indent=2), "utf-8")
    if empty_ledgers:
        for name in psa.LEDGER_FILES:
            (run_dir / name).write_text("", encoding="utf-8")
    else:
        _write_jsonl(
            run_dir / "equity.jsonl",
            [{"decision_bar": bar, "equity": 900000.0 + i, "cash": 937669.47} for i, bar in enumerate(bars)],
        )
        _write_jsonl(run_dir / "targets.jsonl", [{"decision_bar": bars[0], "equity": 1000000.0}])
        _write_jsonl(
            run_dir / "fills.jsonl",
            [{"decision_bar": bars[0], "symbol": "AAA", "qty": 10.0, "fill_price": 1.0}],
        )
        _write_jsonl(
            run_dir / "unfilled.jsonl",
            [{"decision_bar": bars[0], "symbol": "BBB", "residual_qty": -5.0}],
        )
        _write_jsonl(run_dir / "concordance.jsonl", [{"decision_bar": bar, "active_share": 0.01} for bar in bars])
        flags = list(clean) if clean is not None else [True] * len(bars)
        _write_jsonl(
            run_dir / "regime.jsonl",
            [{"decision_bar": bar, "clean": flag, "failures": []} for bar, flag in zip(bars, flags)],
        )
    (run_dir / "nightly.log").write_text(
        nightly if nightly is not None else _nightly_log(run_date=bars[-1]), encoding="utf-8"
    )
    return run_dir


def run_assessment(run_dir: Path, tmp_path: Path, *extra: str, asof: str = ASOF) -> tuple[int, str]:
    """Invoke the CLI offline, writing the report outside the run directory."""
    out = tmp_path / "report.txt"
    code = psa.main([str(run_dir), "--asof", asof, "--out", str(out), *extra])
    return code, out.read_text(encoding="utf-8") if out.exists() else ""


def _snapshot(path: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# the four required shapes


def test_clean_directory_reports_no_gap(tmp_path: Path) -> None:
    """A run directory current through the last closed session has nothing to report as missed."""
    run_dir = build_run_dir(tmp_path, bars=("2026-09-02", "2026-09-03", "2026-09-04"))

    code, report = run_assessment(run_dir, tmp_path)

    assert code == psa.EXIT_OK
    assert "last completed session  2026-09-04" in report
    assert "missed sessions         0" in report
    assert "no session ran" not in report


def test_multi_session_gap_lists_every_missed_session_and_invents_none(tmp_path: Path) -> None:
    """The fifteen sessions after 2026-08-14 are reported as a gap, Labor Day is not one of them."""
    run_dir = build_run_dir(tmp_path)

    code, report = run_assessment(run_dir, tmp_path)

    assert code == psa.EXIT_OK
    assert "missed sessions         15" in report
    listed = [line.split()[0] for line in report.splitlines() if line.strip().endswith("no session ran")]
    assert [line.strip() for line in listed] == list(MISSED_2026)
    # Labor Day is a closure, not a missed session: named, and not in the gap list.
    assert "2026-09-07  Labor Day" in report
    assert "2026-09-07" not in listed
    # Nothing was fabricated for the interruption.
    assert "no session ran" in report
    for missed in MISSED_2026:
        assert f"{missed}  no session ran" in report
    assert "never counted as clean sessions" in report


def test_unknown_state_schema_fails_loud(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unrecognized schema_version stops the assessment; it is never read best-effort."""
    run_dir = build_run_dir(tmp_path, state=_state(schema_version=99))
    out = tmp_path / "report.txt"

    code = psa.main([str(run_dir), "--asof", ASOF, "--out", str(out)])

    assert code == psa.EXIT_UNREADABLE
    assert not out.exists(), "a refused assessment must not leave a report behind"
    err = capsys.readouterr().err
    assert "schema_version 99" in err
    assert "state.py" in err


def test_empty_ledgers_report_empty_rather_than_guessing(tmp_path: Path) -> None:
    """Zero-length ledgers give an unknown last session and no gap — not a fabricated one."""
    run_dir = build_run_dir(tmp_path, empty_ledgers=True)

    code, report = run_assessment(run_dir, tmp_path)

    assert code == psa.EXIT_OK
    assert "last completed session  UNKNOWN" in report
    assert "No gap is computed" in report
    assert "no session ran" not in report
    assert f"0/{psa.CLEAN_SESSIONS_REQUIRED} consecutive clean sessions" in report
    assert "equity.jsonl          0 row(s)" in report


# ---------------------------------------------------------------------------
# the properties that make the report trustworthy


def test_run_directory_is_never_written(tmp_path: Path) -> None:
    """Byte-for-byte and mtime-for-mtime, the run directory is untouched by a full run."""
    run_dir = build_run_dir(tmp_path)
    before = _snapshot(run_dir)

    code, _ = run_assessment(run_dir, tmp_path)

    assert code == psa.EXIT_OK
    assert _snapshot(run_dir) == before


def test_report_may_not_be_written_into_the_run_directory(tmp_path: Path) -> None:
    run_dir = build_run_dir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        psa.main([str(run_dir), "--asof", ASOF, "--out", str(run_dir / "report.txt")])

    assert excinfo.value.code == 2
    assert not (run_dir / "report.txt").exists()


def test_relative_run_dir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The path is an argument precisely so it cannot resolve against the wrong cwd."""
    build_run_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        psa.main(["runs/fixture", "--asof", ASOF])

    assert excinfo.value.code == 2


def test_offline_by_default_and_never_imports_the_broker_chain(tmp_path: Path) -> None:
    """No credential, no flag: the report says the venue was not read, and says so plainly."""
    run_dir = build_run_dir(tmp_path)

    code, report = run_assessment(run_dir, tmp_path)

    assert code == psa.EXIT_OK
    assert "NOT PERFORMED" in report
    assert "No credential was used and no venue was" in report

    module = ast.parse(Path(psa.__file__).read_text(encoding="utf-8"))
    top_level = {
        alias.name if isinstance(node, ast.Import) else node.module or ""
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        name.startswith("prism.live") for name in top_level
    ), f"the broker/config import chain must stay behind --network; module imports {sorted(top_level)}"


def test_offline_run_loads_no_venue_module_and_leaks_no_key(tmp_path: Path) -> None:
    """In a fresh interpreter with credentials set, the offline path loads no venue client and prints no key.

    The in-process check above cannot see this: another test in this file
    imports ``prism.live.state``, so by the time it runs the broker chain may
    already be in ``sys.modules``. This one runs the CLI in a subprocess whose
    only credentials are sentinels, and asserts both halves of the report's
    claim — no venue was contacted, and no key was used — against the loaded
    module set and the report text.
    """
    run_dir = build_run_dir(tmp_path)
    out = tmp_path / "subprocess_report.txt"
    sentinel_id = "SENTINEL-KEY-ID-DO-NOT-LEAK"
    sentinel_secret = "SENTINEL-SECRET-DO-NOT-LEAK"

    probe = (
        "import json, sys\n"
        "from prism.scripts import paper_state_assessment as psa\n"
        f"code = psa.main([{str(run_dir)!r}, '--asof', {ASOF!r}, '--out', {str(out)!r}])\n"
        "print(json.dumps({'code': code, 'modules': sorted(sys.modules)}))\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(psa.__file__).resolve().parents[2]),
        "APCA_API_KEY_ID": sentinel_id,
        "APCA_API_SECRET_KEY": sentinel_secret,
    }
    # -P keeps the cwd off sys.path, so the closure is the package's, not the checkout's.
    proc = subprocess.run(
        [sys.executable, "-P", "-c", probe], capture_output=True, text=True, env=env, cwd=str(tmp_path)
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(proc.stdout.splitlines()[-1])
    assert result["code"] == psa.EXIT_OK
    loaded = set(result["modules"])
    assert "alpaca" not in {name.split(".")[0] for name in loaded}, "an offline assessment loaded the venue SDK"
    assert "prism.live.alpaca" not in loaded
    assert not any(name.startswith("prism.live") for name in loaded), sorted(n for n in loaded if "prism" in n)

    report = out.read_text(encoding="utf-8")
    assert sentinel_id not in report and sentinel_secret not in report
    assert "NOT PERFORMED" in report


def test_accepted_state_schemas_track_the_loader() -> None:
    """If ``prism.live.state`` bumps its schema, this reader's accepted set must be bumped with it."""
    from prism.live.state import STATE_SCHEMA_VERSION

    assert STATE_SCHEMA_VERSION == psa.CURRENT_STATE_SCHEMA_VERSION
    assert STATE_SCHEMA_VERSION in psa.SUPPORTED_STATE_SCHEMA_VERSIONS


# ---------------------------------------------------------------------------
# the regime clock


def test_regime_clock_is_the_doctors_streak(tmp_path: Path) -> None:
    """The reported streak is computed to agree with doctor.check_regime_clock on the same ledger."""
    from prism.scripts.doctor import check_regime_clock

    run_dir = build_run_dir(
        tmp_path,
        bars=("2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"),
        clean=(True, True, False, True, True),
    )

    _, report = run_assessment(run_dir, tmp_path)
    doctor = check_regime_clock(run_dir)

    assert f"2/{psa.CLEAN_SESSIONS_REQUIRED} consecutive clean sessions through 2026-08-14" in report
    assert doctor.detail.startswith(f"2/{psa.CLEAN_SESSIONS_REQUIRED} consecutive clean sessions through 2026-08-14")


def test_gap_neither_advances_nor_resets_the_clock_and_the_policy_stays_undecided(tmp_path: Path) -> None:
    """The clock is the same number before and after fifteen unrun sessions, and nobody decided anything."""
    current = build_run_dir(tmp_path / "current", bars=("2026-09-02", "2026-09-03", "2026-09-04"))
    dark = build_run_dir(tmp_path / "dark")

    _, current_report = run_assessment(current, tmp_path / "current")
    _, dark_report = run_assessment(dark, tmp_path / "dark")

    streak = f"3/{psa.CLEAN_SESSIONS_REQUIRED} consecutive clean sessions"
    assert streak in current_report and streak in dark_report
    assert "policy                  UNDECIDED" in dark_report
    assert "does not answer it" in dark_report


def test_trailing_unclean_row_is_called_out(tmp_path: Path) -> None:
    run_dir = build_run_dir(tmp_path, clean=(True, True, False))

    _, report = run_assessment(run_dir, tmp_path)

    assert f"0/{psa.CLEAN_SESSIONS_REQUIRED} consecutive clean sessions" in report
    assert "The newest row is not clean; the streak was already 0 before the gap." in report


# ---------------------------------------------------------------------------
# ledgers, log, and the disagreements between them


def test_malformed_ledger_line_is_counted_not_silently_dropped(tmp_path: Path) -> None:
    run_dir = build_run_dir(tmp_path)
    with (run_dir / "equity.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"decision_bar": "2026-08-1\n')

    code, report = run_assessment(run_dir, tmp_path)

    assert code == psa.EXIT_OK
    assert "1 MALFORMED LINE(S)" in report


def test_last_wrapper_verdict_is_not_the_doctors_exit(tmp_path: Path) -> None:
    """`EXIT 1` from the wrapper must not be masked by the `DOCTOR EXIT 0` that follows it."""
    run_dir = build_run_dir(tmp_path, nightly=_nightly_log(run_date="2026-08-14", exit_code=1))

    _, report = run_assessment(run_dir, tmp_path)

    assert "last wrapper exit EXIT 1 at 2026-08-14T18:30:27-0700" in report
    assert "15 pass, 1 warn, 0 fail" in report
    assert "WARN  regime-clock" in report


def test_state_and_ledger_disagreement_is_reported_not_resolved(tmp_path: Path) -> None:
    run_dir = build_run_dir(tmp_path, state=_state(pending_decision_bar="2026-08-17"))

    _, report = run_assessment(run_dir, tmp_path)

    assert "DISAGREEMENT" in report
    assert "pending_decision_bar 2026-08-17 != equity.jsonl last mark 2026-08-14" in report
    # The equity ledger still anchors the gap; the disagreement does not silently move it.
    assert "last completed session  2026-08-14" in report


def test_missing_state_json_is_unreadable(tmp_path: Path) -> None:
    run_dir = build_run_dir(tmp_path)
    (run_dir / "state.json").unlink()

    assert psa.main([str(run_dir), "--asof", ASOF]) == psa.EXIT_UNREADABLE


def test_dates_outside_the_session_table_fail_loud(tmp_path: Path) -> None:
    """Rather than counting weekdays past the table's edge, the assessment stops.

    The dates are before the table's start, which stage 2 moved back to
    2020-01-01 and stage 3b (decision D12) moved back again to 2006-01-01 so
    the recorded trend evaluation could decide on its own 2006-2026 sessions.
    The property is the edge, not the year.
    """
    assert dt.date.fromisoformat("2005-12-30") < sessions.coverage()[0]
    run_dir = build_run_dir(tmp_path, bars=("2005-12-28", "2005-12-29", "2005-12-30"))

    assert psa.main([str(run_dir), "--asof", ASOF]) == psa.EXIT_UNREADABLE


def test_operator_note_is_recorded_verbatim(tmp_path: Path) -> None:
    run_dir = build_run_dir(tmp_path)

    _, report = run_assessment(run_dir, tmp_path, "--note", "restart deferred: owner travelling until 09-15")

    assert "NOTE: restart deferred: owner travelling until 09-15" in report


# ---------------------------------------------------------------------------
# the commit question


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_commit_difference_names_the_commits_and_the_paths_they_touch(tmp_path: Path) -> None:
    """The traded commit versus the checkout that would restart the loop, and what moved between them."""
    repo = tmp_path / "repo"
    (repo / "src" / "prism" / "live").mkdir(parents=True)
    (repo / "ops").mkdir()
    (repo / "src" / "prism" / "live" / "loop.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "ops" / "nightly.sh").write_text("echo one\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    traded = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "src" / "prism" / "live" / "loop.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "ops" / "nightly.sh").write_text("echo two\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "move the loop and the wrapper")

    run_dir = build_run_dir(repo, nightly=_nightly_log(run_date="2026-08-14", commit=traded))
    _, report = run_assessment(run_dir, tmp_path)

    assert f"last session traded  {traded}" in report
    assert "commits HEAD adds    1" in report
    assert "move the loop and the wrapper" in report
    assert "src/prism/live/loop.py" in report
    assert "ops/nightly.sh" in report
    assert "A restart runs the current checkout, not the commit the last session traded." in report


def test_commit_difference_is_reported_undetermined_outside_a_checkout(tmp_path: Path) -> None:
    """No repository is a reported fact, not a crash and not a fabricated 'no difference'."""
    run_dir = build_run_dir(tmp_path)

    code, report = run_assessment(run_dir, tmp_path)

    assert code == psa.EXIT_OK
    assert "NOT DETERMINED" in report
    assert "no git checkout contains" in report


# ---------------------------------------------------------------------------
# the session-table health warning


def test_table_expiry_warning_fires_within_ninety_days(tmp_path: Path) -> None:
    from prism.io import sessions

    run_dir = build_run_dir(tmp_path, bars=("2027-10-27", "2027-10-28", "2027-10-29"))
    near = (sessions.coverage_end() - dt.timedelta(days=30)).isoformat()

    _, near_report = run_assessment(run_dir, tmp_path, asof=near)
    _, far_report = run_assessment(run_dir, tmp_path, asof=ASOF)

    assert f"WARNING: the bundled session table ends {sessions.coverage_end().isoformat()}" in near_report
    assert "WARNING: the bundled session table ends" not in far_report


def test_environment_needs_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With every Alpaca variable stripped, the assessment still completes."""
    for key in [k for k in os.environ if k.startswith("APCA")]:
        monkeypatch.delenv(key, raising=False)
    run_dir = build_run_dir(tmp_path)

    assert run_assessment(run_dir, tmp_path)[0] == psa.EXIT_OK
