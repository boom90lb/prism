"""The `prism` command surface (spec §3.14).

Two things are being protected here. The first is the vocabulary: every command
the specification declares parses from the first release, and the ones this
release does not run from the command line say so and name what works instead,
because an operator reading a refusal should learn where to go rather than read
a stack trace. The second is the one operating session the command line does
run — `prism run --once --broker synthetic` — whose refusals (a live account, a
real broker, a scheduled stage) are the ones that must never quietly succeed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path

import pytest

import prism
from prism import cli
from prism.cli import EXIT_LATER_STAGE, EXIT_REFUSED, build_parser, main
from prism.live.broker import Broker
from prism.template import write_template

#: The phrase every command-line refusal of an unavailable path carries.
NOT_AVAILABLE = "is not available from the command line in this release: "

#: Internal references a user cannot look up: delivery stages of the engine
#: overhaul, sections of its unpublished implementation specification, and its
#: numbered design questions. None may reach a user-visible message.
INTERNAL_REFERENCE = re.compile(r"\bstage \d|§\s?3\.|\[Q\d")

# argv, and a fragment of what the refusal must name as working instead.
DEFERRED_COMMANDS = [
    (["data", "fetch", "--source", "twelvedata", "--symbols", "u.txt"], "prism data ingest"),
    (["data", "manifest", "--freeze"], "manifest.json"),
    (["data", "status"], "manifest.json"),
    (["data", "universe", "--index", "sp500"], "prism-build-universe"),
    (["compare", "a.json", "b.json"], "prism.validation.compare.compare_reevaluation"),
    (["compare", "--registration", "2026-09-construction"], "prism.validation.compare.compare_reevaluation"),
    (["trials", "reconcile"], "prism.validation.trials.reconcile_trial_log"),
    (["folds", "import", "--from", "folds.json", "--out", "folds.toml"], "FoldTable.from_records"),
    (["reconcile", "runs/paper"], "prism-doctor --network"),
    (["check", "runs/paper", "--monitor"], "prism.scripts.paper_monitor"),
]

#: `prism replay` and `prism upgrade-diff` are delivered: both drive
#: `prism.live.replay.upgrade_diff_report` over recorded captures. Their exit
#: codes and refusals are tested at the end of this file, and the rows above
#: no longer name them, so a stale refusal cannot return without this list
#: changing.

#: `prism check` runs (freshness and the missed-run monitor); only its
#: `--monitor` statistics are still refused, which is the row above.
#: Its behaviour is tested in tests/test_freshness.py, where the session table
#: that answers "was that a missed run or a holiday?" is under test too.

#: The two data commands the command line runs, each refusing for its own
#: reason rather than as unavailable. They are exercised here so the deferred
#: list above cannot quietly regrow them.
DELIVERED_DATA_COMMANDS = [
    ["data", "ingest", "--format", "csv", "--mapping", "m.toml", "bars.csv"],
    ["data", "import-legacy", "--cache-dir", "no_such_cache"],
]

DEFERRED_EVALUATE_FLAGS = [
    ([], "prism.evaluate.walk_forward_evaluate"),  # no --smoke at all
    (["--smoke", "--engine", "canonical"], "CanonicalSettings"),
    (["--smoke", "--registration", "r"], "walk_forward_evaluate(registration="),
    (["--smoke", "--arm", "decile_control"], "walk_forward_evaluate(registration="),
    (["--smoke", "--visibility", "replay_recorded"], "visibility_mode='replay_recorded'"),
    (["--smoke", "--vintage", "v1"], "--workspace"),
    (["--smoke", "--freeze-scores", "panel.parquet"], "check_freeze_record"),
]

DEFERRED_RUN_FLAGS = [
    # `--broker alpaca` with no stage: the decide stage constructs the book over
    # a live dataset view, which this command does not do.
    ["--account", "paper", "--once", "--broker", "alpaca"],
    ["--account", "paper", "--once", "--broker", "alpaca", "--stage", "decide"],
    ["--account", "paper", "--once", "--stage", "decide"],
    ["--account", "paper"],  # no --once: the resident loop
]


@pytest.fixture(autouse=True, scope="module")
def _restore_logging():
    """Undo the file handlers a command installs on the root logger."""
    before = list(logging.getLogger().handlers)
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A written project with a small synthetic fixture, shared by the session tests."""
    root = tmp_path_factory.mktemp("cli") / "proj"
    write_template(root)
    assert (
        main(
            [
                "--workspace",
                str(root),
                "data",
                "synthetic",
                "--seed",
                "3",
                "--scenarios",
                "membership,costs,rounding,distribution",
                "--equities",
                "12",
                "--etfs",
                "2",
            ]
        )
        == 0
    )
    return root


# --------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------


def test_every_declared_command_parses() -> None:
    # The whole surface of the §3.14 table, so a command never dies on
    # `invalid choice` when the specification says it exists.
    declared = {
        "init",
        "data",
        "conform",
        "evaluate",
        "compare",
        "trials",
        "folds",
        "run",
        "reconcile",
        "replay",
        "check",
        "upgrade-diff",
    }
    actions = [a for a in build_parser()._actions if a.choices and a.dest == "command"]
    assert declared == set(actions[0].choices)


def _ids(value: object) -> str:
    return " ".join(value) if isinstance(value, list) else str(value)


@pytest.mark.parametrize("argv,instead", DEFERRED_COMMANDS, ids=_ids)
def test_a_deferred_command_names_what_works_instead(argv: list[str], instead: str, capsys) -> None:
    assert main(argv) == EXIT_LATER_STAGE
    message = capsys.readouterr().err
    assert NOT_AVAILABLE in message, message
    assert instead in message.split(NOT_AVAILABLE, 1)[1], message
    assert not INTERNAL_REFERENCE.search(message), message
    assert "Traceback" not in message


@pytest.mark.parametrize("argv", DELIVERED_DATA_COMMANDS, ids=lambda v: " ".join(v))
def test_a_delivered_data_command_refuses_for_its_own_reason_not_as_unavailable(
    argv: list[str], capsys, tmp_path, monkeypatch
) -> None:
    """``data ingest`` and ``data import-legacy`` run from the command line."""
    # Resolve the workspace (logs/, .env) to a scratch directory, never the checkout.
    monkeypatch.delenv("PRISM_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(argv) == EXIT_REFUSED
    message = capsys.readouterr().err
    assert NOT_AVAILABLE not in message, message
    assert "Traceback" not in message


@pytest.mark.parametrize("flags,instead", DEFERRED_EVALUATE_FLAGS, ids=_ids)
def test_evaluate_refuses_what_the_smoke_run_cannot_record(flags: list[str], instead: str, capsys) -> None:
    assert main(["evaluate", "specs/book.toml", *flags]) == EXIT_LATER_STAGE
    message = capsys.readouterr().err
    assert NOT_AVAILABLE in message and instead in message, message
    assert not INTERNAL_REFERENCE.search(message), message


@pytest.mark.parametrize("flags", DEFERRED_RUN_FLAGS, ids=lambda v: " ".join(v))
def test_run_refuses_anything_but_one_synthetic_session(flags: list[str], capsys) -> None:
    assert main(["run", "--spec", "specs/book.toml", *flags]) == EXIT_LATER_STAGE
    message = capsys.readouterr().err
    assert NOT_AVAILABLE in message, message
    assert not INTERNAL_REFERENCE.search(message), message


def test_run_requires_an_account(capsys) -> None:
    # `--account` is mandatory and has no default [Q5]: argparse refuses.
    with pytest.raises(SystemExit) as exit_info:
        main(["run", "--spec", "specs/book.toml", "--once"])
    assert exit_info.value.code == 2
    assert "--account" in capsys.readouterr().err


def test_version_reports_the_renamed_distribution(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"prism-trading {prism.__version__}"


def test_no_command_prints_the_surface(capsys) -> None:
    assert main([]) == EXIT_LATER_STAGE
    assert "upgrade-diff" in capsys.readouterr().out


def test_data_without_a_subcommand_says_what_to_try(capsys) -> None:
    assert main(["data"]) == EXIT_REFUSED
    assert "prism data synthetic" in capsys.readouterr().err


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def test_init_writes_a_project_and_refuses_to_overwrite_it(tmp_path: Path, capsys) -> None:
    root = tmp_path / "my-strategy"
    assert main(["init", str(root)]) == 0
    assert (root / "specs" / "book.toml").is_file()
    assert f"prism-trading=={prism.__version__}" in capsys.readouterr().out
    assert main(["init", str(root)]) == EXIT_REFUSED
    assert "already holds" in capsys.readouterr().err


def test_init_creates_no_workspace_beside_it(tmp_path: Path, monkeypatch) -> None:
    # `prism init` resolves no workspace: running it must not scatter data/,
    # results/ and state/ across whatever directory the operator stood in.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    assert main(["init", "proj"]) == 0
    assert sorted(p.name for p in cwd.iterdir()) == ["proj"]


# --------------------------------------------------------------------------
# data synthetic
# --------------------------------------------------------------------------


def test_data_synthetic_writes_a_manifest(project: Path, capsys) -> None:
    assert (project / "data" / "manifest.json").is_file()


def test_data_synthetic_refuses_a_bad_scenario_list(tmp_path: Path, capsys) -> None:
    root = tmp_path / "proj"
    write_template(root)
    assert main(["--workspace", str(root), "data", "synthetic", "--seed", "1", "--scenarios", "all,split"]) == (
        EXIT_REFUSED
    )
    assert "cannot be combined" in capsys.readouterr().err
    assert main(["--workspace", str(root), "data", "synthetic", "--seed", "1", "--scenarios", "no_such"]) == (
        EXIT_REFUSED
    )
    assert "unknown scenarios" in capsys.readouterr().err
    assert not (root / "data" / "manifest.json").exists()


# --------------------------------------------------------------------------
# run --once against the synthetic broker
# --------------------------------------------------------------------------


def test_run_without_a_dataset_names_the_fixture_command(tmp_path: Path, capsys) -> None:
    root = tmp_path / "proj"
    write_template(root)
    argv = ["--workspace", str(root), "run", "--spec", str(root / "specs" / "book.toml")]
    assert main([*argv, "--account", "paper", "--once", "--broker", "synthetic"]) == EXIT_REFUSED
    assert "prism data synthetic" in capsys.readouterr().err


def test_run_once_decides_one_session_and_refuses_the_second(project: Path, capsys) -> None:
    from prism.live.state import StateStore

    argv = [
        "--workspace",
        str(project),
        "run",
        "--spec",
        str(project / "specs" / "book.toml"),
        "--account",
        "paper",
        "--once",
        "--broker",
        "synthetic",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "synthetic broker" in out
    assert "decile_neutral" in out

    # A synthetic session declares the account id `synthetic`, so it can never
    # contend with a real account's lock, and it writes state under it.
    run_dir = project / "state" / "synthetic"
    state = StateStore(run_dir / "state.json").load()
    assert state is not None
    assert state.pending_decision_bar == "2027-12-31"
    assert state.pending_orders, "the decided orders are written ahead of submission"
    assert state.schema_version == 3, "the loop writes LoopState at the version the tree carries"
    assert (run_dir / ".lock").exists()

    # The replay stream may never resume or be mistaken for a live ledger.
    assert main(argv) == EXIT_REFUSED
    assert "already holds loop state" in capsys.readouterr().err


def test_run_refuses_a_rule_the_live_cycle_cannot_construct(project: Path, capsys) -> None:
    argv = [
        "--workspace",
        str(project),
        "run",
        "--spec",
        str(project / "specs" / "etf_core.toml"),
        "--account",
        "paper",
        "--once",
        "--broker",
        "synthetic",
    ]
    assert main(argv) == EXIT_LATER_STAGE
    message = capsys.readouterr().err
    assert "top_k_equal_weight" in message and NOT_AVAILABLE in message and "--smoke" in message
    assert not INTERNAL_REFERENCE.search(message), message


def test_run_refuses_an_unloadable_specification(project: Path, tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text((project / "specs" / "book.toml").read_text().replace('path = "strategies', 'path = "nowhere'))
    argv = [
        "--workspace",
        str(project),
        "run",
        "--spec",
        str(broken),
        "--account",
        "paper",
        "--once",
        "--broker",
        "synthetic",
    ]
    assert main(argv) == EXIT_REFUSED
    message = capsys.readouterr().err
    assert "nowhere" in message and "Traceback" not in message


# --------------------------------------------------------------------------
# The book configuration a session is run under
# --------------------------------------------------------------------------


def test_the_binding_cap_is_the_smaller_of_the_rule_and_the_profile(project: Path, tmp_path: Path) -> None:
    from prism.cli import _daily_book_config
    from prism.spec import load_spec
    from prism.workspace import Workspace

    workspace = Workspace(project)
    spec = load_spec(project / "specs" / "book.toml", workspace=workspace)
    config = _daily_book_config(spec, spec.components[0])
    assert config.book == "decile_neutral"
    assert config.decile == pytest.approx(0.10)
    assert config.max_gross == pytest.approx(1.0)
    assert config.max_symbol_abs_weight == pytest.approx(0.10)
    assert config.whole_shares is True, "the synthetic execution profile allows no fractional quantity"
    assert config.max_participation is None, "the paper profile declares no participation limit"
    assert config.decision_every == 21, "the component's update cadence is the refresh cadence"

    # A profile is a limit, not a suggestion: a gross bound below the rule's own
    # cap binds, because caps are down-only.
    tighter = tmp_path / "tight"
    tighter.mkdir()
    for name in ("specs", "profiles", "costs", "policies", "strategies"):
        (tighter / name).mkdir()
    for source in project.rglob("*.toml"):
        target = tighter / source.relative_to(project)
        if target.parent.exists():
            target.write_text(source.read_text())
    (tighter / "strategies" / "my_signal.py").write_text((project / "strategies" / "my_signal.py").read_text())
    profile = tighter / "profiles" / "paper.toml"
    profile.write_text(profile.read_text().replace("value = 1.0", "value = 0.25"))
    tight_spec = load_spec(tighter / "specs" / "book.toml", workspace=Workspace(tighter))
    assert _daily_book_config(tight_spec, tight_spec.components[0]).max_gross == pytest.approx(0.25)


def test_refusal_messages_never_end_in_a_traceback(capsys) -> None:
    # Every message this file asserts on is printed, not raised: a refusal is a
    # sentence, and only a defect is a traceback.
    for argv, _instead in DEFERRED_COMMANDS:
        main(argv)
    captured = capsys.readouterr().err
    assert "Traceback" not in captured
    assert captured.count(NOT_AVAILABLE) == len(DEFERRED_COMMANDS)


def _every_parser(parser: argparse.ArgumentParser):
    """The parser and every subcommand parser beneath it, depth first."""
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                yield from _every_parser(child)


def test_help_text_cites_no_internal_reference() -> None:
    # `--help` is user-visible output: it may not send a reader to a stage of
    # the engine overhaul or to a section of a specification that is not published.
    for parser in _every_parser(build_parser()):
        text = parser.format_help()
        assert not INTERNAL_REFERENCE.search(text), (parser.prog, INTERNAL_REFERENCE.search(text))


# --------------------------------------------------------------------------
# run --broker alpaca: the authorized venue session (§3.10, §3.14)
#
# Every refusal here is reached through `main`, the way an operator reaches it,
# and against a broker double that fails the test on any call it did not have
# to make. That is what separates "the control is wired" from "a gate a test
# built refuses when a test calls it": the entry point loads the record, builds
# the identity from what this process actually is, and verifies the binding
# before the session can do anything.
# --------------------------------------------------------------------------


class VenueDouble(Broker):
    """A venue that records every call and fails the test on any order path."""

    def __init__(self, account_id: str = "PA-CLI-0001", endpoint: str | None = None) -> None:
        from prism.live.authorization import PAPER_BASE_URL

        self.endpoint = PAPER_BASE_URL if endpoint is None else endpoint
        self.calls: list[str] = []
        self.gate = None
        self._account_id = account_id

    def bind_authorization(self, gate) -> None:
        self.gate = gate

    def account_identity(self):
        from prism.live.broker import AccountIdentity

        self.calls.append("account_identity")
        return AccountIdentity(account_id=self._account_id, endpoint=self.endpoint, is_paper=True)

    def _reached(self, name: str) -> None:
        self.calls.append(name)
        raise AssertionError(f"the venue was reached: {name}. The refusal must precede every request (§3.10)")

    def positions(self) -> dict[str, float]:
        self._reached("positions")
        return {}

    def cash(self) -> float:
        self._reached("cash")
        return 0.0

    def submit(self, order) -> None:
        self._reached("submit")

    def submitted_order_ids(self, since: str | None = None) -> set[str]:
        self._reached("submitted_order_ids")
        return set()

    def fills_for(self, client_order_ids: set[str]):
        self._reached("fills_for")
        return []


@pytest.fixture()
def venue(monkeypatch):
    """The construction seam, wired to a double that must never be traded through."""
    double = VenueDouble()
    built: list[str] = []

    def factory(mode: str, *, time_in_force: str = "opg"):
        built.append(mode)
        return double

    monkeypatch.setattr(cli, "venue_broker", factory)
    double.built = built  # type: ignore[attr-defined]
    return double


def _write_authorization(project: Path, account: str, **overrides) -> Path:
    """The §3.10 record for ``account``, built from what this process really is."""
    import socket

    from prism.live.authorization import (
        AUTHORIZATION_FILENAME,
        PAPER_BASE_URL,
        distribution_id,
    )
    from prism.spec import engine_version, load_spec
    from prism.workspace import resolve_workspace

    workspace = resolve_workspace(str(project))
    spec = load_spec(project / "specs" / "book.toml", workspace=workspace)
    payload: dict = {
        "schema": 1,
        "mode": "paper",
        "account_name": "reference-paper",
        "account_id": "PA-CLI-0001",
        "endpoint": PAPER_BASE_URL,
        "execution_host": socket.gethostname(),
        "config_id": spec.config_id(),
        "engine_version": engine_version(),
        "distribution_id": distribution_id(),
        "policy_ids": {"profile": "p", "costs": "c", "execution": "e", "monitoring": "m", "recovery": "r"},
        "scope": {"add_exposure": True, "manage_existing": True, "hedge": False},
        "submission_window_sessions": 1,
        "granted_by": "owner",
        "granted_at": "2026-09-09T00:00:00+00:00",
        "expires_at": None,
    }
    payload.update(overrides)
    run_dir = workspace.run_dir(account)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / AUTHORIZATION_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_argv(project: Path, *flags: str) -> list[str]:
    return [
        "--workspace",
        str(project),
        "run",
        "--spec",
        str(project / "specs" / "book.toml"),
        *flags,
    ]


def test_a_live_session_with_no_record_is_refused_at_the_entry_point(project: Path, venue, capsys) -> None:
    """S4b through `main`: no record, no client, no credential, no venue call."""
    from prism.workspace import resolve_workspace

    record = resolve_workspace(str(project)).run_dir("live") / "authorization.json"
    assert not record.exists()

    code = main(_run_argv(project, "--account", "live", "--broker", "alpaca", "--once", "--stage", "submit"))
    assert code == EXIT_REFUSED
    message = capsys.readouterr().err
    assert "no authorization record" in message and "S4b" in message
    assert not INTERNAL_REFERENCE.search(message), message
    # The double was never built and never called: the refusal precedes the client.
    assert venue.built == [] and venue.calls == []


def test_a_venue_session_refuses_a_record_that_names_another_host(project: Path, venue, capsys) -> None:
    """Everything else in the record is this process's own truth; the host is not."""
    _write_authorization(project, "paper", execution_host="some-other-host")
    code = main(_run_argv(project, "--account", "paper", "--broker", "alpaca", "--once", "--stage", "submit"))
    assert code == EXIT_REFUSED
    message = capsys.readouterr().err
    assert "execution_host=" in message and "some-other-host" in message
    # The account id was read once, to bind against; nothing else was asked.
    assert venue.calls == ["account_identity"]


def test_a_venue_session_refuses_a_record_written_for_another_account(project: Path, venue, capsys) -> None:
    _write_authorization(project, "paper", account_id="PA-SOMEONE-ELSE")
    code = main(_run_argv(project, "--account", "paper", "--broker", "alpaca", "--once", "--stage", "submit"))
    assert code == EXIT_REFUSED
    assert "account_id=" in capsys.readouterr().err
    assert venue.calls == ["account_identity"]


def test_a_venue_session_refuses_a_record_granted_for_another_configuration(project: Path, venue, capsys) -> None:
    """`config_id` comes from the specification this process resolved (§3.10)."""
    _write_authorization(project, "paper", config_id="cfg-" + "9" * 8)
    code = main(_run_argv(project, "--account", "paper", "--broker", "alpaca", "--once", "--stage", "submit"))
    assert code == EXIT_REFUSED
    assert "config_id=" in capsys.readouterr().err


def test_a_bound_venue_session_takes_the_account_lock(project: Path, venue, capsys) -> None:
    """The whole chain, end to end: record, identity, binding, lock, stage.

    Nothing is pending, so the submit stage has nothing to offer — which is the
    point: the controls ran, the account-keyed lock was taken from the record,
    and no order path was reached.
    """
    from prism.live.lockfile import account_lock_path, read_lock_record

    _write_authorization(project, "paper")
    code = main(_run_argv(project, "--account", "paper", "--broker", "alpaca", "--once", "--stage", "submit"))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "account reference-paper (PA-CLI-0001)" in out
    assert "nothing pending" in out
    assert venue.built == ["paper"] and venue.calls == ["account_identity"]
    assert venue.gate is not None and venue.gate.record.account_id == "PA-CLI-0001"
    # The lock the record declares was taken on this host, and released.
    lock = account_lock_path("PA-CLI-0001")
    assert lock.is_file(), lock
    assert read_lock_record(lock).released_at is not None


# --------------------------------------------------------------------------
# run --break-lock (§3.10)
# --------------------------------------------------------------------------


def test_break_lock_requires_a_reason(project: Path, capsys) -> None:
    code = main(_run_argv(project, "--account", "paper", "--once", "--break-lock"))
    assert code == EXIT_REFUSED
    assert "requires `--reason`" in capsys.readouterr().err


def test_break_lock_records_who_broke_it_when_and_why(project: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    """The run-directory record is written before the lock changes."""
    import socket

    from prism.live import lockfile
    from prism.live.lockfile import (
        account_execution_lock,
        account_lock_path,
        read_lock_record,
    )
    from prism.workspace import resolve_workspace

    monkeypatch.setattr(lockfile, "default_lock_dir", lambda: tmp_path)
    path = account_lock_path("synthetic")
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "account",
                "key": "synthetic",
                "host": "some-other-host",
                "pid": 4321,
                "start_token": None,
                "acquired_at": "2026-09-09T00:00:00+00:00",
                "heartbeat_at": "2026-09-09T00:00:00+00:00",
                "released_at": None,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    code = main(
        _run_argv(
            project,
            "--account",
            "paper",
            "--once",
            "--break-lock",
            "--reason",
            "the declared host was rebuilt and its lock outlived it",
        )
    )
    assert code == 0
    assert "broke the execution lock" in capsys.readouterr().out

    ledger = resolve_workspace(str(project)).run_dir("synthetic") / "lock_breaks.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["reason"] == "the declared host was rebuilt and its lock outlived it"
    assert row["broken_by"].endswith(f"@{socket.gethostname()}:{os.getpid()}")
    assert row["account_id"] == "synthetic" and row["displaced"]["host"] == "some-other-host"
    assert row["broken_at"] <= read_lock_record(path).released_at

    # The break took effect: the lock is takeable again on this host.
    with account_execution_lock("synthetic", socket.gethostname(), lock_dir=tmp_path):
        pass


def test_break_lock_on_a_venue_account_needs_the_record_that_names_it(project: Path, capsys) -> None:
    code = main(
        _run_argv(project, "--account", "live", "--broker", "alpaca", "--once", "--break-lock", "--reason", "x")
    )
    assert code == EXIT_REFUSED
    assert "no authorization record" in capsys.readouterr().err


# --------------------------------------------------------------------------
# `prism replay` and `prism upgrade-diff`: the recorded-session upgrade review
# --------------------------------------------------------------------------

#: The stage 0 capture. Recorded sessions with no bar panel, which is exactly
#: the state in which an upgrade review has to say what it cannot answer.
CAPTURE = Path(__file__).parent / "fixtures" / "paper_replay"


def _capture_with_a_changed_weight(destination: Path) -> Path:
    """A copy of the capture whose second decision moved one name by 1%."""
    shutil.copytree(CAPTURE, destination)
    targets = destination / "targets.jsonl"
    rows = [json.loads(line) for line in targets.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = rows[-1]
    victim = sorted(row["targets"])[0]
    row["targets"][victim] = float(row["targets"][victim]) + 0.01
    targets.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return destination


@pytest.mark.skipif(not (CAPTURE / "manifest.json").is_file(), reason="the stage 0 capture is not in the tree")
def test_upgrade_diff_over_the_fixture_compares_nothing_and_exits_incomplete(tmp_path: Path, capsys) -> None:
    """No candidate capture, so nothing is compared and nothing is claimed.

    Exit 2 under the `prism.live.parity_gate` convention: nothing failed and
    something could not be evaluated. Exit 0 here would read as "the upgrade
    changes nothing", which is the one thing this report cannot say.
    """
    out_json = tmp_path / "report.json"
    assert main(["upgrade-diff", "--sessions", str(CAPTURE), "--json", str(out_json)]) == EXIT_LATER_STAGE
    out = capsys.readouterr().out
    assert "0/5 session(s) compared" in out
    assert "does not say the upgrade changes nothing" in out
    # The capture's own caveat about the missing panels reaches the operator.
    assert "NO PANELS" in out
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["sessions_compared"] == 0 and report["sessions_requested"] == 5
    assert report["complete"] is False


@pytest.mark.skipif(not (CAPTURE / "manifest.json").is_file(), reason="the stage 0 capture is not in the tree")
def test_upgrade_diff_of_the_capture_against_itself_is_a_clean_exit(capsys) -> None:
    argv = ["upgrade-diff", "--sessions", str(CAPTURE), "--candidate", str(CAPTURE)]
    assert main(argv) == 0
    assert "5/5 session(s) compared, 0 with a different decision" in capsys.readouterr().out


@pytest.mark.skipif(not (CAPTURE / "manifest.json").is_file(), reason="the stage 0 capture is not in the tree")
def test_a_changed_decision_exits_one_and_names_the_session(tmp_path: Path, capsys) -> None:
    candidate = _capture_with_a_changed_weight(tmp_path / "candidate")
    argv = ["upgrade-diff", "--sessions", str(CAPTURE), "--candidate", str(candidate)]
    assert main(argv) == EXIT_REFUSED
    out = capsys.readouterr().out
    assert "5/5 session(s) compared, 1 with a different decision" in out
    assert "2026-08-11" in out


@pytest.mark.skipif(not (CAPTURE / "manifest.json").is_file(), reason="the stage 0 capture is not in the tree")
def test_one_bar_can_be_reviewed_on_its_own(capsys) -> None:
    argv = ["upgrade-diff", "--sessions", str(CAPTURE), "--candidate", str(CAPTURE), "--bar", "2026-07-13"]
    assert main(argv) == 0
    assert "1/1 session(s) compared" in capsys.readouterr().out


@pytest.mark.skipif(not (CAPTURE / "manifest.json").is_file(), reason="the stage 0 capture is not in the tree")
def test_replay_upgrade_diff_is_the_same_report(capsys) -> None:
    assert main(["replay", str(CAPTURE), "--upgrade-diff", "--candidate", str(CAPTURE)]) == 0
    assert "5/5 session(s) compared" in capsys.readouterr().out


def test_replay_without_the_diff_flag_names_the_tool_that_drives_a_cycle(capsys) -> None:
    assert main(["replay", "runs/paper"]) == EXIT_REFUSED
    err = capsys.readouterr().err
    assert "prism.scripts.replay_loop" in err
    assert not INTERNAL_REFERENCE.search(err), err


def test_a_capture_without_a_manifest_is_a_refusal_not_a_traceback(tmp_path: Path, capsys) -> None:
    (tmp_path / "empty").mkdir()
    assert main(["upgrade-diff", "--sessions", str(tmp_path / "empty")]) == EXIT_REFUSED
    err = capsys.readouterr().err
    assert "holds no manifest.json" in err
    assert "Traceback" not in err
