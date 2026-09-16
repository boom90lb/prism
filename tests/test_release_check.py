"""``ops/release_check.py``: the checks a release must pass before anyone depends on it.

The script itself shells out to ``uv`` and builds real artifacts, which is a
minute of work and a network round trip; these tests exercise everything that
decides a verdict without either. What is pinned here is the arithmetic that
turns findings into an exit code, the artifact readers, and the two properties
the script exists to guarantee: it refuses a credential-shaped path, and it has
no publish path at all.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "ops" / "release_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is absent for a module loaded by path.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rc = _load()


# ---------------------------------------------------------------------------
# fixtures: artifacts that look like the real ones without building anything
# ---------------------------------------------------------------------------

METADATA = """Metadata-Version: 2.5
Name: prism-trading
Version: 0.4.0
Requires-Python: >=3.12
Requires-Dist: numpy>=1.24.0
Requires-Dist: pandas>=2.0.0
Provides-Extra: construction
Requires-Dist: cvxpy>=1.5.0; extra == 'construction'
Requires-Dist: clarabel>=0.9.0; extra == 'construction'

The description.
"""


def make_wheel(directory: Path, *, extra_members: dict[str, str] | None = None) -> Path:
    path = directory / "prism_trading-0.4.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("prism/__init__.py", "__version__ = '0.4.0'\n")
        archive.writestr("prism_trading-0.4.0.dist-info/METADATA", METADATA)
        for name, body in (extra_members or {}).items():
            archive.writestr(name, body)
    return path


def make_sdist(directory: Path, *, extra_members: dict[str, str] | None = None) -> Path:
    path = directory / "prism_trading-0.4.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, body in {"prism_trading-0.4.0/PKG-INFO": METADATA, **(extra_members or {})}.items():
            payload = body.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def write_pyproject(directory: Path, *, name: str = "prism-trading", version: str = "0.4.0") -> None:
    (directory / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8")


# ---------------------------------------------------------------------------
# the verdict arithmetic
# ---------------------------------------------------------------------------


def test_every_check_passing_exits_zero() -> None:
    checks = [rc.Check("a", rc.PASS, ""), rc.Check("b", rc.PASS, "")]
    assert rc.exit_code(checks) == rc.EXIT_OK
    assert rc.verdict_word(rc.exit_code(checks)) == "PASS"


def test_an_unevaluated_check_is_incomplete_and_never_a_pass() -> None:
    """The whole point of the third verdict: unrunnable is not clean."""
    checks = [rc.Check("a", rc.PASS, ""), rc.Check("b", rc.NOT_EVALUATED, "no index")]
    assert rc.exit_code(checks) == rc.EXIT_INCOMPLETE
    assert rc.verdict_word(rc.exit_code(checks)) == "INCOMPLETE"


def test_a_failure_outranks_an_unevaluated_check() -> None:
    checks = [rc.Check("a", rc.FAIL, "bad"), rc.Check("b", rc.NOT_EVALUATED, "no index")]
    assert rc.exit_code(checks) == rc.EXIT_FAILED


# ---------------------------------------------------------------------------
# credential shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "member",
    [
        ".env",
        "prism_trading-0.4.0/.env",
        "prism/.ENV",
        "deploy/id_rsa",
        "certs/client.pem",
        "certs/CLIENT.PEM",
        "a/b/secrets.toml",
        "keys/broker.key",
    ],
)
def test_a_credential_shaped_path_is_found_anywhere_in_the_tree(member: str) -> None:
    assert rc.credential_shaped([member]) == [member]


@pytest.mark.parametrize(
    "member",
    ["prism/cli.py", "docs/security.md", "data/universe/sp500_current.txt", "prism/live/keyed_lock.py"],
)
def test_ordinary_paths_are_not_flagged(member: str) -> None:
    assert rc.credential_shaped([member]) == []


def test_contents_refuses_an_artifact_carrying_a_dotenv(tmp_path: Path) -> None:
    wheel = make_wheel(tmp_path, extra_members={"prism/.env": "APCA_API_SECRET_KEY=x\n"})
    sdist = make_sdist(tmp_path)
    check = rc.check_contents(wheel, sdist)
    assert check.verdict == rc.FAIL
    assert "prism/.env" in check.detail


def test_contents_passes_a_clean_pair(tmp_path: Path) -> None:
    check = rc.check_contents(make_wheel(tmp_path), make_sdist(tmp_path))
    assert check.verdict == rc.PASS
    assert check.facts["wheel_top_level"] == ["prism", "prism_trading-0.4.0.dist-info"]


# ---------------------------------------------------------------------------
# reading the artifacts
# ---------------------------------------------------------------------------


def test_wheel_metadata_is_read_off_the_artifact_not_the_source(tmp_path: Path) -> None:
    metadata = rc.wheel_metadata(make_wheel(tmp_path))
    assert metadata["name"] == "prism-trading"
    assert metadata["version"] == "0.4.0"
    assert metadata["requires_python"] == ">=3.12"
    assert metadata["core_requirements"] == ["numpy>=1.24.0", "pandas>=2.0.0"]
    assert metadata["extra_requirements"] == {"construction": ["clarabel>=0.9.0", "cvxpy>=1.5.0"]}


def test_artifact_facts_hash_both_kinds(tmp_path: Path) -> None:
    make_wheel(tmp_path)
    make_sdist(tmp_path)
    (tmp_path / "notes.txt").write_text("not a distribution", encoding="utf-8")
    facts = rc.artifact_facts(tmp_path)
    assert set(facts) == {"prism_trading-0.4.0-py3-none-any.whl", "prism_trading-0.4.0.tar.gz"}
    for row in facts.values():
        assert len(row["sha256"]) == 64
        assert row["bytes"] > 0


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "payload.bin"
    path.write_bytes(b"prism" * 1000)
    assert rc.sha256_file(path) == hashlib.sha256(b"prism" * 1000).hexdigest()


def test_artifact_identity_refuses_a_version_the_source_does_not_declare(tmp_path: Path) -> None:
    write_pyproject(tmp_path, version="0.5.0")
    check, metadata = rc.check_artifact_identity(tmp_path, make_wheel(tmp_path))
    assert check.verdict == rc.FAIL
    assert "0.5.0" in check.detail and "0.4.0" in check.detail
    assert metadata["version"] == "0.4.0"


def test_artifact_identity_passes_when_the_wheel_and_the_source_agree(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    check, _ = rc.check_artifact_identity(tmp_path, make_wheel(tmp_path))
    assert check.verdict == rc.PASS


# ---------------------------------------------------------------------------
# requirement names and the lockfile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("numpy>=1.24.0", "numpy"),
        ("python-dotenv>=1.0.0", "python-dotenv"),
        ("python_dotenv>=1.0.0", "python-dotenv"),
        ("jax[cuda12]>=0.5.3", "jax"),
        ("statsmodels>=0.14.0; extra == 'ensemble'", "statsmodels"),
        ("requests (>=2.30.0)", "requests"),
        ("scipy", "scipy"),
    ],
)
def test_requirement_name_strips_markers_extras_and_specifiers(requirement: str, expected: str) -> None:
    assert rc.requirement_name(requirement) == expected


def test_lock_entries_read_version_source_and_sdist_hash(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        "\n".join(
            [
                "[[package]]",
                'name = "numpy"',
                'version = "2.1.0"',
                'source = { registry = "https://pypi.org/simple" }',
                'sdist = { url = "https://files.pythonhosted.org/x.tar.gz", hash = "sha256:abc" }',
                "wheels = [",
                '    { url = "https://files.pythonhosted.org/x.whl", hash = "sha256:def" },',
                "]",
                "",
                "[[package]]",
                'name = "prism_trading"',
                'version = "0.4.0"',
                'source = { editable = "." }',
            ]
        ),
        encoding="utf-8",
    )
    entries = rc.lock_entries(lock)
    assert entries["numpy"]["version"] == "2.1.0"
    assert entries["numpy"]["sdist_sha256"] == "sha256:abc"
    assert entries["numpy"]["n_wheels"] == 1
    # The lock spells the project with an underscore; the check normalises.
    assert rc.lock_describes_project(entries, "prism-trading", "0.4.0")
    assert not rc.lock_describes_project(entries, "prism-trading", "0.5.0")
    assert not rc.lock_describes_project(entries, "prism", "0.4.0")


def test_the_repositorys_own_lockfile_parses() -> None:
    """Whatever it currently says, it must be readable; the verdict is the script's job."""
    entries = rc.lock_entries(REPO_ROOT / "uv.lock")
    assert len(entries) > 50
    assert "numpy" in entries


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_a_redirected_index_fails_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A release resolved against another index is a release about that index."""
    monkeypatch.setenv("UV_INDEX_URL", "https://internal.example/simple")
    write_pyproject(tmp_path)
    metadata = rc.wheel_metadata(make_wheel(tmp_path))
    check = rc.check_dependency_provenance(tmp_path, metadata, rc.Check("fresh-install", rc.PASS, "", {}))
    assert check.verdict == rc.FAIL
    assert "UV_INDEX_URL" in check.detail
    assert check.facts["index"]["redirected"] is True


def test_a_stale_lockfile_is_incomplete_and_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in rc.INDEX_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    write_pyproject(tmp_path)
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "prism"\nversion = "0.3.4"\nsource = { editable = "." }\n', encoding="utf-8"
    )
    metadata = rc.wheel_metadata(make_wheel(tmp_path))
    install = rc.Check(
        "fresh-install",
        rc.PASS,
        "",
        {
            "offline": False,
            "installed": {
                "prism-trading": {"version": "0.4.0", "record_sha256": "a", "installer": "uv"},
                "numpy": {"version": "2.1.0", "record_sha256": "b", "installer": "uv"},
                "pandas": {"version": "2.2.0", "record_sha256": "c", "installer": "uv"},
            },
        },
    )
    check = rc.check_dependency_provenance(tmp_path, metadata, install)
    assert check.verdict == rc.NOT_EVALUATED
    assert "does not describe prism-trading 0.4.0" in check.detail
    assert check.facts["lock"]["describes_this_release"] is False


def test_a_declared_requirement_missing_from_the_closure_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in rc.INDEX_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    write_pyproject(tmp_path)
    metadata = rc.wheel_metadata(make_wheel(tmp_path))
    install = rc.Check(
        "fresh-install",
        rc.PASS,
        "",
        {
            "offline": False,
            "installed": {
                "prism-trading": {"version": "0.4.0", "record_sha256": "a", "installer": "uv"},
                "numpy": {"version": "2.1.0", "record_sha256": "b", "installer": "uv"},
            },
        },
    )
    check = rc.check_dependency_provenance(tmp_path, metadata, install)
    assert check.verdict == rc.FAIL
    assert "pandas" in check.detail


# ---------------------------------------------------------------------------
# the property the script exists to keep
# ---------------------------------------------------------------------------


def test_the_script_has_no_publish_path() -> None:
    """Publishing is an owner act. A check that could upload is a check that will.

    Guarded on the source text rather than on behaviour, because the failure
    this prevents is somebody adding an upload step later.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("uv publish", "twine", "upload_dist", "--publish-url", "PyPI token", "PYPI_TOKEN"):
        assert forbidden not in source, f"the release check must not be able to publish; found {forbidden!r}"


def test_the_rendered_report_says_it_does_not_publish() -> None:
    report = {
        "distribution": "prism-trading",
        "version": "0.4.0",
        "source_stamp": "git:abc",
        "verdict": "PASS",
        "exit_code": 0,
        "checks": [{"name": "build", "verdict": "PASS", "detail": "built", "facts": {}}],
        "artifacts": {"prism_trading-0.4.0-py3-none-any.whl": {"sha256": "0" * 64, "bytes": 1}},
    }
    text = rc.render(report)
    assert "never publishes" in text
    assert "0" * 64 in text
    assert json.dumps(report)  # the report the artifact carries is JSON-serialisable
