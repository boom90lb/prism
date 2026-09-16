"""§3.10 at the two legacy paper entry points (`paper_loop`, `paper_sweep`).

These two shells are what runs today, and stage 5 does not take that away: with
no authorization record in the run directory they behave exactly as before,
which is the whole of the backward-compatibility requirement. What stage 5 adds
is that neither of them can be pointed at a live account, and neither of them
can run *beside* a record it is unable to bind against.

Both refusals are proved through ``main`` — the way the nightly wrapper reaches
them — and both are proved to happen before a credential is read: every test
here deletes the Alpaca variables from the environment first and changes into a
directory with no ``.env``, so a refusal that did not fire could not construct a
client at all. Nothing here opens a socket.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from prism.live.authorization import (
    AUTHORIZATION_FILENAME,
    LIVE_BASE_URL,
    PAPER_BASE_URL,
    AuthorizationRefused,
)
from prism.scripts import paper_loop, paper_sweep

ENTRY_POINTS = {"loop": paper_loop, "sweep": paper_sweep}


@pytest.fixture()
def offline(tmp_path: Path, monkeypatch) -> Path:
    """A run directory, no credentials in the environment, and no `.env` above it."""
    monkeypatch.chdir(tmp_path)
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "runs" / "paper"
    run_dir.mkdir(parents=True)
    return run_dir


def _argv(which: str, run_dir: Path) -> list[str]:
    if which == "loop":
        return ["--run-dir", str(run_dir), "--symbols", "AAA,BBB"]
    return ["--run-dir", str(run_dir)]


@pytest.mark.parametrize("which", sorted(ENTRY_POINTS))
def test_a_live_endpoint_is_refused_before_a_credential_is_read(which: str, offline: Path, monkeypatch) -> None:
    """The mode is an argument, never an inheritance: these shells are paper.

    An exported ``APCA_API_BASE_URL`` may set an endpoint but never the mode, so
    a live endpoint under these entry points is refused where the endpoint is
    resolved — before the credentials are read, which is what makes it provable
    that no request could have been sent.
    """
    monkeypatch.setenv("APCA_API_BASE_URL", LIVE_BASE_URL)
    with pytest.raises(AuthorizationRefused, match="live endpoint under mode 'paper'"):
        ENTRY_POINTS[which].main(_argv(which, offline))


@pytest.mark.parametrize("which", sorted(ENTRY_POINTS))
def test_a_paper_endpoint_is_not_refused(which: str, offline: Path, monkeypatch) -> None:
    """The control refuses the live endpoint and nothing else.

    With the paper endpoint the entry point proceeds past §3.10 and stops at the
    missing credentials, which is exactly where it stopped before this change.
    """
    monkeypatch.setenv("APCA_API_BASE_URL", PAPER_BASE_URL)
    with pytest.raises(RuntimeError) as refusal:
        ENTRY_POINTS[which].main(_argv(which, offline))
    assert "APCA_API_KEY_ID" in str(refusal.value)


@pytest.mark.parametrize("which", sorted(ENTRY_POINTS))
def test_a_record_the_entry_point_cannot_bind_against_is_refused(which: str, offline: Path) -> None:
    """A record beside a shell that computes no `config_id` is refused, not honoured."""
    (offline / AUTHORIZATION_FILENAME).write_text(
        json.dumps(
            {
                "schema": 1,
                "mode": "paper",
                "account_name": "reference-paper",
                "account_id": "PA-0001",
                "endpoint": PAPER_BASE_URL,
                "execution_host": socket.gethostname(),
                "config_id": "cfg-0",
                "engine_version": "0.4.0",
                "distribution_id": "git:" + "a" * 40,
                "policy_ids": {"profile": "p", "costs": "c", "execution": "e", "monitoring": "m", "recovery": "r"},
                "scope": {"add_exposure": True, "manage_existing": True, "hedge": False},
                "submission_window_sessions": 1,
                "granted_by": "owner",
                "granted_at": "2026-09-09T00:00:00+00:00",
                "expires_at": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(AuthorizationRefused, match="cannot bind against it"):
        ENTRY_POINTS[which].main(_argv(which, offline))


@pytest.mark.parametrize("which", sorted(ENTRY_POINTS))
def test_no_record_is_not_an_error(which: str, offline: Path, monkeypatch) -> None:
    """The pre-authorization paper instrument: absence is a fact, not a failure.

    This is the backward-compatibility assertion. With no record the entry point
    runs on past §3.10 into the credential read it has always done — the same
    place the paper loop reaches today.
    """
    monkeypatch.setenv("APCA_API_BASE_URL", PAPER_BASE_URL)
    assert not (offline / AUTHORIZATION_FILENAME).exists()
    with pytest.raises(RuntimeError) as refusal:
        ENTRY_POINTS[which].main(_argv(which, offline))
    assert "authorization" not in str(refusal.value).lower()
