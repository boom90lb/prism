#!/usr/bin/env python3
"""Check a release before anyone depends on it: build it, hash it, and say where its dependencies come from.

This is the runnable half of ``docs/depending_on_prism.md``. It answers three
questions a user cannot answer by trusting a version string:

1. What are the exact bytes of this release? (``build``, ``reproducible``,
   ``artifact-identity``, ``contents``)
2. Does the artifact install and run on its own, away from the checkout that
   built it? (``fresh-install``)
3. What does it pull in, from where, and does the repository's lockfile still
   describe it? (``dependency-provenance``)

It never publishes. There is no upload path in this file and adding one is an
owner act, not a check: uploading is what makes a name and a version permanent
for everyone, and a script that can do it by accident is a script that will.

Verdicts and the exit code follow ``prism.live.parity_gate``: 0 every check
passed, 1 a check failed, 2 a check could not be evaluated and none failed.
"NOT EVALUATED" is a first-class verdict, never folded into a pass, because
"the wheel's dependencies were never resolved" and "the wheel's dependencies
resolved cleanly" are the two answers a release review must not confuse.

Usage::

    UV_CACHE_DIR="$TMPDIR/uv-cache" .venv/bin/python ops/release_check.py
    UV_CACHE_DIR="$TMPDIR/uv-cache" .venv/bin/python ops/release_check.py --offline
    .venv/bin/python ops/release_check.py --out docs/release_check.json

Stdlib only, plus ``uv`` on PATH. It imports nothing from ``prism``, so it runs
against a checkout whose engine does not import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Iterable, Sequence

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUATED = "NOT EVALUATED"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

#: Path shapes that must never reach an index. Matched against every member
#: name of both artifacts, case-folded. A release that ships one of these has
#: published a secret, and no later revocation un-publishes it.
CREDENTIAL_SHAPES: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".netrc",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "secrets.toml",
    "secrets.yaml",
    "secrets.yml",
)

#: Suffixes that carry key material whatever the basename is.
CREDENTIAL_SUFFIXES: tuple[str, ...] = (".pem", ".key", ".p12", ".pfx", ".keystore")

#: The index a default `uv` install reads. Recorded rather than assumed, and
#: compared with the environment, because "which index served this" is the
#: provenance question a hash alone does not answer.
DEFAULT_INDEX = "https://pypi.org/simple"

#: Environment variables that redirect resolution away from the default index.
INDEX_ENV_VARS: tuple[str, ...] = (
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_INDEX",
    "UV_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One named check, its verdict, and the facts it established."""

    name: str
    verdict: str
    detail: str
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "verdict": self.verdict, "detail": self.detail, "facts": self.facts}


def exit_code(checks: Sequence[Check]) -> int:
    """0 all passed, 1 any failed, 2 any unevaluated with nothing failed.

    A failure outranks an unevaluated check: a release that failed a check it
    could run is refused whether or not some other check was unrunnable.
    """
    if any(check.verdict == FAIL for check in checks):
        return EXIT_FAILED
    if any(check.verdict == NOT_EVALUATED for check in checks):
        return EXIT_INCOMPLETE
    return EXIT_OK


def verdict_word(code: int) -> str:
    return {EXIT_OK: "PASS", EXIT_FAILED: "FAIL", EXIT_INCOMPLETE: "INCOMPLETE"}[code]


# ---------------------------------------------------------------------------
# Artifact facts
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_facts(directory: Path) -> dict[str, dict[str, Any]]:
    """``{filename: {sha256, bytes}}`` for every distribution file in ``directory``."""
    facts: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return facts
    for path in sorted(directory.iterdir()):
        if path.name.endswith(".whl") or path.name.endswith(".tar.gz"):
            facts[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return facts


def wheel_metadata(wheel: Path) -> dict[str, Any]:
    """Name, version, requires-python and Requires-Dist read from the wheel itself.

    Read from the artifact and never from ``pyproject.toml``: the point of the
    check is that the built bytes declare what the source says they declare.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{wheel.name}: expected exactly one dist-info/METADATA, found {len(metadata_names)}")
        payload = archive.read(metadata_names[0])
    message = BytesParser().parsebytes(payload)
    requires = [str(value) for value in message.get_all("Requires-Dist") or ()]
    core = sorted(r for r in requires if "extra ==" not in r)
    extras = {}
    for extra in sorted(str(v) for v in message.get_all("Provides-Extra") or ()):
        marker = f"extra == '{extra}'"
        extras[extra] = sorted(r.split(";")[0].strip() for r in requires if marker in r)
    return {
        "name": str(message.get("Name") or ""),
        "version": str(message.get("Version") or ""),
        "requires_python": str(message.get("Requires-Python") or ""),
        "core_requirements": core,
        "extra_requirements": extras,
    }


def archive_members(path: Path) -> list[str]:
    """Every member path inside a wheel or an sdist, without the leading root."""
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            return sorted(archive.namelist())
    with tarfile.open(path) as archive:
        return sorted(name.split("/", 1)[1] if "/" in name else name for name in archive.getnames())


def credential_shaped(members: Iterable[str]) -> list[str]:
    """Members whose name says they carry a secret, whatever they actually hold.

    Shape only. A release check cannot read every file's meaning, and it does
    not have to: a distribution has no legitimate reason to carry a path with
    one of these names, so the shape is the finding.
    """
    hits = []
    for member in members:
        base = member.rsplit("/", 1)[-1].casefold()
        if base in CREDENTIAL_SHAPES or base.endswith(CREDENTIAL_SUFFIXES):
            hits.append(member)
    return sorted(hits)


# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------


def run(command: Sequence[str], *, cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def uv_build(root: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return run(["uv", "build", "--out-dir", str(out_dir)], cwd=root)


def index_provenance() -> dict[str, Any]:
    """Which index resolution will read, and whether anything redirected it."""
    overrides = {name: os.environ[name] for name in INDEX_ENV_VARS if os.environ.get(name)}
    return {
        "default_index": DEFAULT_INDEX,
        "environment_overrides": overrides,
        "redirected": bool(overrides),
    }


INSTALLED_REPORT = r"""
import hashlib, importlib.metadata as md, json
rows = {}
for dist in md.distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    try:
        record = dist.read_text("RECORD")
    except Exception:
        record = None
    try:
        raw = dist.read_text("direct_url.json")
        direct = json.loads(raw) if raw else None
    except Exception:
        direct = None
    try:
        installer = (dist.read_text("INSTALLER") or "").strip() or None
    except Exception:
        installer = None
    rows[name] = {
        "version": dist.version,
        "installer": installer,
        "record_sha256": None if record is None else hashlib.sha256(record.encode("utf-8")).hexdigest(),
        "record_entries": None if record is None else len(record.strip().splitlines()),
        "direct_url": direct,
    }
print(json.dumps(rows))
"""


def installed_report(python: Path) -> dict[str, Any]:
    """Every distribution in ``python``'s environment, with its installed-content digest.

    ``record_sha256`` is the sha256 of the installed distribution's ``RECORD``,
    which is the same identity ``prism.live.authorization.distribution_id``
    computes for the engine when it runs from an installed wheel rather than a
    git checkout. It fingerprints what was installed, not what an index served,
    and the two are different facts.
    """
    completed = run([str(python), "-c", INSTALLED_REPORT])
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "the installed-distribution report failed")
    return json.loads(completed.stdout)


# ---------------------------------------------------------------------------
# The lockfile cross-check
# ---------------------------------------------------------------------------


def lock_entries(lock_path: Path) -> dict[str, dict[str, Any]]:
    """``{name: {version, source, sdist_url, sdist_sha256, n_wheels}}`` from ``uv.lock``."""
    payload = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = {}
    for package in payload.get("package", []):
        name = str(package.get("name", ""))
        if not name:
            continue
        sdist = package.get("sdist") or {}
        entries[name.replace("_", "-").casefold()] = {
            "version": str(package.get("version", "")),
            "source": package.get("source"),
            "sdist_url": sdist.get("url"),
            "sdist_sha256": sdist.get("hash"),
            "n_wheels": len(package.get("wheels") or []),
        }
    return entries


def requirement_name(requirement: str) -> str:
    """The bare distribution name of a ``Requires-Dist`` string, normalised."""
    head = requirement.split(";", 1)[0].strip()
    for stop in ("[", "(", "=", "<", ">", "!", "~", " "):
        index = head.find(stop)
        if index != -1:
            head = head[:index]
    return head.strip().replace("_", "-").casefold()


def lock_describes_project(lock: dict[str, dict[str, Any]], name: str, version: str) -> bool:
    """Whether ``uv.lock`` still names this project at this version.

    A lockfile that names an older distribution or an older version is not a
    lockfile for this release, and every dependency row read out of it is a row
    about some other build.
    """
    entry = lock.get(name.replace("_", "-").casefold())
    return entry is not None and entry.get("version") == version


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def check_build(root: Path, out_dir: Path) -> tuple[Check, dict[str, dict[str, Any]]]:
    completed = uv_build(root, out_dir)
    if completed.returncode != 0:
        return (
            Check("build", FAIL, f"uv build exited {completed.returncode}: {completed.stderr.strip()[:400]}"),
            {},
        )
    facts = artifact_facts(out_dir)
    wheels = [name for name in facts if name.endswith(".whl")]
    sdists = [name for name in facts if name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        return (
            Check(
                "build",
                FAIL,
                f"expected one wheel and one sdist, built {len(wheels)} wheel(s) and {len(sdists)} sdist(s)",
                {"artifacts": facts},
            ),
            facts,
        )
    return (
        Check("build", PASS, f"built {sdists[0]} and {wheels[0]}", {"artifacts": facts}),
        facts,
    )


def check_reproducible(root: Path, first: dict[str, dict[str, Any]], second_dir: Path) -> Check:
    completed = uv_build(root, second_dir)
    if completed.returncode != 0:
        return Check(
            "reproducible",
            NOT_EVALUATED,
            f"the second build exited {completed.returncode} so the two builds cannot be compared",
        )
    second = artifact_facts(second_dir)
    differing = sorted(
        name
        for name in set(first) | set(second)
        if first.get(name, {}).get("sha256") != second.get(name, {}).get("sha256")
    )
    if differing:
        return Check(
            "reproducible",
            FAIL,
            f"{len(differing)} artifact(s) differ between two builds of the same tree: {', '.join(differing)}. "
            "The recorded hash is then a fact about one build, not about the source",
            {"differing": differing, "second_build": second},
        )
    return Check(
        "reproducible",
        PASS,
        f"{len(first)} artifact(s) hash identically across two builds of the same tree, "
        "so the recorded hashes are a property of the source and not of the build run",
        {"second_build": second},
    )


def check_artifact_identity(root: Path, wheel: Path) -> tuple[Check, dict[str, Any]]:
    metadata = wheel_metadata(wheel)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared_name = str(project.get("name", ""))
    declared_version = str(project.get("version", ""))
    mismatches = []
    if metadata["name"] != declared_name:
        mismatches.append(f"wheel Name={metadata['name']!r}, pyproject name={declared_name!r}")
    if metadata["version"] != declared_version:
        mismatches.append(f"wheel Version={metadata['version']!r}, pyproject version={declared_version!r}")
    filename_stem = wheel.name.split("-")[0]
    if filename_stem.replace("_", "-").casefold() != declared_name.replace("_", "-").casefold():
        mismatches.append(f"wheel filename names {filename_stem!r}, pyproject names {declared_name!r}")
    facts = {
        "wheel_metadata": metadata,
        "pyproject_name": declared_name,
        "pyproject_version": declared_version,
    }
    if mismatches:
        return Check("artifact-identity", FAIL, "; ".join(mismatches), facts), metadata
    return (
        Check(
            "artifact-identity",
            PASS,
            f"{metadata['name']} {metadata['version']}, requires-python {metadata['requires_python']}, "
            f"{len(metadata['core_requirements'])} core requirement(s) and "
            f"{len(metadata['extra_requirements'])} extra(s), all read off the built wheel",
            facts,
        ),
        metadata,
    )


def check_contents(wheel: Path, sdist: Path) -> Check:
    wheel_members = archive_members(wheel)
    sdist_members = archive_members(sdist)
    hits = {
        wheel.name: credential_shaped(wheel_members),
        sdist.name: credential_shaped(sdist_members),
    }
    tops = sorted({name.split("/")[0] for name in wheel_members})
    facts = {
        "wheel_entries": len(wheel_members),
        "sdist_entries": len(sdist_members),
        "wheel_top_level": tops,
        "credential_shaped": {k: v for k, v in hits.items() if v},
    }
    flagged = sorted(name for names in hits.values() for name in names)
    if flagged:
        return Check(
            "contents",
            FAIL,
            f"{len(flagged)} credential-shaped path(s) in the artifacts: {', '.join(flagged[:8])}",
            facts,
        )
    return Check(
        "contents",
        PASS,
        f"wheel {len(wheel_members)} entries under {', '.join(tops)}; sdist {len(sdist_members)} entries; "
        "no credential-shaped path in either",
        facts,
    )


def check_fresh_install(wheel: Path, venv_dir: Path, *, offline: bool, expected_version: str) -> Check:
    created = run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    if created.returncode != 0:
        return Check(
            "fresh-install",
            NOT_EVALUATED,
            f"uv venv exited {created.returncode}: {created.stderr.strip()[:300]}",
        )
    python = venv_dir / "bin" / "python"
    command = ["uv", "pip", "install", "--python", str(python)]
    if offline:
        command += ["--offline", "--no-deps"]
    command.append(str(wheel))
    installed = run(command)
    if installed.returncode != 0:
        return Check(
            "fresh-install",
            NOT_EVALUATED if offline else FAIL,
            f"uv pip install exited {installed.returncode}: {installed.stderr.strip()[:400]}",
        )
    try:
        report = installed_report(python)
    except RuntimeError as exc:
        return Check("fresh-install", NOT_EVALUATED, f"the installed-distribution report failed: {exc}")
    engine = report.get("prism-trading") or report.get("prism_trading")
    if engine is None:
        return Check("fresh-install", FAIL, "the wheel installed but no prism-trading distribution is present")
    facts = {
        "engine_version": engine["version"],
        "distribution_id": None if engine["record_sha256"] is None else "sha256:" + engine["record_sha256"],
        "n_distributions": len(report),
        "installed": report,
        "offline": offline,
    }
    if engine["version"] != expected_version:
        return Check(
            "fresh-install",
            FAIL,
            f"installed version {engine['version']!r} is not the wheel's {expected_version!r}",
            facts,
        )
    if offline:
        return Check(
            "fresh-install",
            NOT_EVALUATED,
            f"installed {engine['version']} with --no-deps under --offline: the import and console-script "
            "checks need the dependency closure, so they were not run",
            facts,
        )
    smoke = run(
        [
            str(python),
            "-c",
            "import prism; from prism.spec import engine_version; print(engine_version())",
        ]
    )
    facts["import_stdout"] = smoke.stdout.strip()
    facts["import_returncode"] = smoke.returncode
    if smoke.returncode != 0:
        return Check(
            "fresh-install",
            FAIL,
            f"the installed wheel does not import: {smoke.stderr.strip()[:400]}",
            facts,
        )
    console = run([str(venv_dir / "bin" / "prism"), "--version"])
    facts["console_returncode"] = console.returncode
    facts["console_stdout"] = console.stdout.strip()
    if console.returncode != 0:
        return Check(
            "fresh-install",
            FAIL,
            f"the `prism` console script exited {console.returncode}: {console.stderr.strip()[:300]}",
            facts,
        )
    return Check(
        "fresh-install",
        PASS,
        f"the wheel installs into a fresh environment ({facts['n_distributions']} distributions), "
        f"imports, reports engine_version {smoke.stdout.strip()}, and its console script runs",
        facts,
    )


def check_dependency_provenance(root: Path, metadata: dict[str, Any], install_check: Check) -> Check:
    lock_path = root / "uv.lock"
    provenance = index_provenance()
    facts: dict[str, Any] = {
        "index": provenance,
        "declared_core": metadata["core_requirements"],
        "declared_extras": metadata["extra_requirements"],
    }
    notes: list[str] = []
    verdict = PASS

    if provenance["redirected"]:
        verdict = FAIL
        notes.append(
            "resolution is redirected away from the default index by "
            + ", ".join(sorted(provenance["environment_overrides"]))
            + "; a release checked against another index is a release about that index"
        )

    installed = install_check.facts.get("installed") or {}
    if installed:
        closure = {
            name: {"version": row["version"], "record_sha256": row["record_sha256"], "installer": row["installer"]}
            for name, row in sorted(installed.items())
            if name.replace("_", "-").casefold() != "prism-trading"
        }
        facts["resolved_closure"] = closure
        declared = {requirement_name(r) for r in metadata["core_requirements"]}
        present = {name.replace("_", "-").casefold() for name in closure}
        missing = sorted(declared - present)
        facts["declared_but_absent"] = missing
        facts["transitive"] = sorted(present - declared)
        if missing and not install_check.facts.get("offline"):
            verdict = FAIL
            notes.append(f"declared requirement(s) absent from the resolved closure: {', '.join(missing)}")
        else:
            notes.append(
                f"{len(closure)} distribution(s) resolved for the core set: "
                f"{len(declared & present)} declared and {len(present - declared)} transitive, "
                "each with the sha256 of its installed RECORD"
            )
    else:
        verdict = NOT_EVALUATED if verdict == PASS else verdict
        notes.append("no fresh environment was produced, so the resolved closure is unknown")

    if not lock_path.is_file():
        facts["lock"] = {"present": False}
        notes.append("no uv.lock in the checkout, so no pinned resolution is recorded here")
        verdict = NOT_EVALUATED if verdict == PASS else verdict
    else:
        entries = lock_entries(lock_path)
        describes = lock_describes_project(entries, metadata["name"], metadata["version"])
        rows = {}
        for requirement in metadata["core_requirements"]:
            name = requirement_name(requirement)
            entry = entries.get(name)
            rows[name] = (
                {"in_lock": False}
                if entry is None
                else {
                    "in_lock": True,
                    "version": entry["version"],
                    "source": entry["source"],
                    "sdist_url": entry["sdist_url"],
                    "sdist_sha256": entry["sdist_sha256"],
                    "n_wheels": entry["n_wheels"],
                }
            )
        facts["lock"] = {
            "present": True,
            "path": str(lock_path.relative_to(root)),
            "n_packages": len(entries),
            "describes_this_release": describes,
            "core_requirements": rows,
        }
        if not describes:
            verdict = NOT_EVALUATED if verdict == PASS else verdict
            notes.append(
                f"uv.lock does not describe {metadata['name']} {metadata['version']}, so every dependency row "
                "read out of it is a row about a different build; `uv lock` refreshes it and needs the index"
            )
        else:
            notes.append(f"uv.lock describes {metadata['name']} {metadata['version']} over {len(entries)} packages")

    return Check("dependency-provenance", verdict, "; ".join(notes), facts)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def render(report: dict[str, Any]) -> str:
    lines = [
        f"release-check: {report['distribution']} {report['version']}",
        f"  source: {report['source_stamp']}",
        f"  verdict: {report['verdict']} (exit {report['exit_code']})",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"  {check['verdict']:<14} {check['name']}")
        lines.append(f"                 {check['detail']}")
    artifacts = report.get("artifacts") or {}
    if artifacts:
        lines.append("")
        lines.append("  artifacts (sha256, bytes):")
        for name, row in artifacts.items():
            lines.append(f"    {row['sha256']}  {row['bytes']:>9}  {name}")
    lines.append("")
    lines.append("  This script never publishes. Uploading a distribution is an owner act.")
    return "\n".join(lines)


def source_stamp(root: Path) -> str:
    head = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if head.returncode != 0 or not head.stdout.strip():
        return "not a git checkout"
    status = run(["git", "-C", str(root), "status", "--porcelain", "-uno"])
    dirty = status.returncode == 0 and bool(status.stdout.strip())
    return f"git:{head.stdout.strip()}" + ("+dirty" if dirty else "")


def release_check(root: Path, *, offline: bool, keep: Path | None = None) -> dict[str, Any]:
    """Run every check and return the report. Writes nothing outside its work directory."""
    checks: list[Check] = []
    work = Path(tempfile.mkdtemp(prefix="release-check-", dir=os.environ.get("TMPDIR") or None))
    artifacts: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {"name": "", "version": "", "core_requirements": [], "extra_requirements": {}}
    try:
        build_check, artifacts = check_build(root, work / "build-a")
        checks.append(build_check)
        if build_check.verdict == PASS:
            wheel = next(work.joinpath("build-a").glob("*.whl"))
            sdist = next(work.joinpath("build-a").glob("*.tar.gz"))
            checks.append(check_reproducible(root, artifacts, work / "build-b"))
            identity, metadata = check_artifact_identity(root, wheel)
            checks.append(identity)
            checks.append(check_contents(wheel, sdist))
            install = check_fresh_install(wheel, work / "venv", offline=offline, expected_version=metadata["version"])
            checks.append(install)
            checks.append(check_dependency_provenance(root, metadata, install))
            if keep is not None:
                keep.mkdir(parents=True, exist_ok=True)
                for path in work.joinpath("build-a").iterdir():
                    shutil.copy2(path, keep / path.name)
        else:
            for name in ("reproducible", "artifact-identity", "contents", "fresh-install", "dependency-provenance"):
                checks.append(Check(name, NOT_EVALUATED, "the build failed, so there is no artifact to check"))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    code = exit_code(checks)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "distribution": metadata["name"] or "unknown",
        "version": metadata["version"] or "unknown",
        "source_stamp": source_stamp(root),
        "root": str(root),
        "offline": offline,
        "publishes": False,
        "artifacts": artifacts,
        "checks": [check.to_dict() for check in checks],
        "verdict": verdict_word(code),
        "exit_code": code,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a release, hash it, and report its dependency provenance.")
    parser.add_argument("--root", default=None, help="the checkout to check (default: this script's repository)")
    parser.add_argument("--out", default=None, help="write the JSON report here as well as to stdout")
    parser.add_argument(
        "--keep-artifacts",
        default=None,
        help="copy the built wheel and sdist here (they are otherwise discarded with the work directory)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not reach the index: the wheel installs with --no-deps and the import, console-script and "
        "closure checks report NOT EVALUATED rather than passing",
    )
    parser.add_argument("--json", action="store_true", help="print the JSON report instead of the table")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    if shutil.which("uv") is None:
        print("release-check: uv is not on PATH; this check builds with uv", file=sys.stderr)
        return EXIT_INCOMPLETE

    report = release_check(root, offline=args.offline, keep=Path(args.keep_artifacts) if args.keep_artifacts else None)
    print(json.dumps(report, indent=2) if args.json else render(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
