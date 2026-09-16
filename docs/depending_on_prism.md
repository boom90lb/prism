# Depending on a release: pins, lockfiles, and how to check a release instead of trusting it

Date: 2026-09-09. Companion to `docs/versioning.md`, which says what a version number promises.
This one is about the practical act of putting `prism-trading` in a private project's dependency
list and knowing what you got.

---

## 1. Pin exactly, and pin the engine separately from your strategy

```toml
# your private project's pyproject.toml
dependencies = [
    "prism-trading==0.4.0",
]
```

Exact, with `==`, not `>=`. The reason is in `docs/versioning.md` section 1: the version contract
covers interfaces, not decisions, so a floor pin lets a release that changes what your book
decides arrive as a background event on somebody's next `uv sync`. An exact pin makes the upgrade
an act you perform, on a day you chose, with the comparison in front of you.

Pin the engine and version your strategy configuration separately. An engine release does not
change or promote your configuration, and your configuration change is not an engine upgrade. The
two identities are already distinct in every record the engine writes: `engine_version` and
`config_id` are separate fields of the authorization record and of every result.

**Distribution name and import name differ.** The distribution is `prism-trading`; the import is
`import prism`. Version 0.4.0 is where the rename landed. The plain name `prism` on PyPI is a
different, unrelated project.

**Nothing is published yet.** As of 2026-09-09, `https://pypi.org/pypi/prism-trading/json`
returns HTTP 404 and `https://pypi.org/pypi/prism/json` returns HTTP 200 (Run 2026-09-09).
Under the normalization rule that treats underscores and hyphens alike, `prism_trading` is the
same name and also returns 404. A 404 establishes only that nothing is published under the name
today. It is not a reservation, and the name is not held until a first
upload succeeds. Until then, install from a local artifact:

```sh
uv build
uv pip install --find-links <checkout>/dist "prism-trading==0.4.0"
```

---

## 2. What a lockfile buys, and what it does not

The project's tool is `uv`, and `uv.lock` plus `uv sync --frozen` is the pinning mechanism. Lock
your private project. Then be precise about what you have bought.

**What pinning buys.**

- The same resolved versions on every machine, every CI run and every rebuild, so a failure that
  appears on one machine is not a resolution difference.
- A name for what you ran. A result records `engine_version`; the lock records the rest of the
  environment. Together they identify the code that produced a number, which is the difference
  between a reproducible result and an anecdote.
- Upgrades become deliberate. The diff of a lockfile is a review artifact: it shows exactly which
  packages moved and by how much, before anything runs.
- Hash-checked installs, where the installer verifies the artifact it fetched against the digest
  the lock recorded, so a substituted artifact fails at install time rather than at runtime.

**What pinning does not buy, stated plainly: a lockfile does not make dependencies safe.** It
pins *which* code you install. It says nothing about *what that code does*. Specifically:

- It is not an audit. Nobody read the pinned code. A lock is a decision to install the same thing
  repeatedly, not a judgement that the thing is good.
- It does not detect a malicious or defective release that you pinned. If the compromise was
  already in the version you locked, the lock reproduces it faithfully, on every machine, for as
  long as the pin stands.
- It freezes known vulnerabilities exactly as well as it freezes known-good code. A pinned
  project does not stop being exposed; it stops being surprised.
- It says nothing about provenance upstream of the artifact: the build machine, the maintainer's
  account, the index. A digest proves you got the same bytes as last time, not that those bytes
  were built from the source they claim.
- Transitive scope is wide. Section 4 below reports the real number for this engine's core set:
  six declared requirements resolve to twelve distributions, so twice as much code as you named
  is installed, and the lock pins all of it without distinguishing what you chose from what you
  inherited.

Pin because you want changes to be deliberate and reviewable. Do not pin and then reason as if
the pinned set had been vetted.

**Keep the lock honest.** A lock that no longer describes the project is worse than no lock,
because it reads as a pinned environment while pinning a different build. This repository's own
`uv.lock` was in that state on 2026-09-09: it named `prism` 0.3.4 and
`uv export --frozen` refused with `Missing workspace member prism-trading` (Run).
The refreshed lock now names `prism-trading` 0.4.0. Tested 2026-09-15:
`uv lock --check` passes, and the release check at commit `f9a8a3a` passes
dependency provenance over 209 locked packages and twelve installed core dependencies.
The earlier INCOMPLETE receipt in section 4 is retained as history.

---

## 3. Keep evaluation environments away from live credentials

Three rules, from the launch brief Q20 and enforced in the engine's own layout:

- Secrets stay outside source. `.env` is gitignored; the rules are in `docs/security.md`.
- No import-time side effects. Importing `prism` reaches nothing and reads no credential.
- Separate environments. The environment you evaluate a private signal in should not carry live
  API keys. The engine's own controls assume this: the mode defaults to paper, `--account
  paper|live` is mandatory with no default, and the endpoint allowlist refuses a live endpoint
  under paper mode before any request
  (`src/prism/live/authorization.py::resolve_endpoint`, `::AuthorizationGate.verify_binding`).

---

## 4. The release check: `ops/release_check.py`

A release you can check is worth more than a release you are told to trust. The check builds the
distribution, records the artifact hashes, installs the result away from the checkout, and
reports the dependency set with its provenance.

```sh
export UV_CACHE_DIR="$TMPDIR/uv-cache"     # uv needs a writable cache
.venv/bin/python ops/release_check.py                       # full check, reaches the index
.venv/bin/python ops/release_check.py --offline             # no index; closure checks report NOT EVALUATED
.venv/bin/python ops/release_check.py --out docs/release_check.json --keep-artifacts dist/
```

Exit 0 every check passed, 1 a check failed, 2 a check could not be evaluated and none failed.
The verdicts follow `prism.live.parity_gate`: NOT EVALUATED is a first-class verdict and is never
folded into a pass.

**It does not publish.** There is no upload path in the script, and
`tests/test_release_check.py::test_the_script_has_no_publish_path` asserts on the source text
that none is added later. Uploading a distribution is an owner act: it makes a name and a version
permanent for everyone, and it is not something a check should be able to do by accident.

### The six checks

| Check | What it establishes |
|---|---|
| `build` | `uv build` produces exactly one sdist and one wheel, with the sha256 and byte size of each. |
| `reproducible` | A second build of the same tree produces byte-identical artifacts, so the recorded hash is a property of the source rather than of one build run. |
| `artifact-identity` | The wheel's own `METADATA` name and version, read out of the artifact rather than out of `pyproject.toml`, agree with the source and with the wheel's filename. |
| `contents` | The member list of both artifacts, and no credential-shaped path in either (`.env`, `id_rsa`, `*.pem`, `*.key`, `secrets.*` and similar, at any depth). A published secret cannot be un-published. |
| `fresh-install` | The wheel installs into a new virtual environment away from the checkout, imports, reports its `engine_version`, and its `prism` console script runs. Records the installed `distribution_id`, which is `sha256:` of the installed `RECORD`, the same identity `prism.live.authorization.distribution_id` computes at run time. |
| `dependency-provenance` | Which index resolution read, and a FAIL if an environment variable redirected it; the declared `Requires-Dist` set per extra, read off the wheel; the resolved closure with each distribution's version and installed-content digest; and whether `uv.lock` still describes this release. |

### Historical check before the lock refresh

Run 2026-09-09 against `git:0cb46b8a40314b64c833bac0aaef717ae6564ff0`, exit 2:

```
release-check: prism-trading 0.4.0
  source: git:0cb46b8a40314b64c833bac0aaef717ae6564ff0
  verdict: INCOMPLETE (exit 2)

  PASS           build
                 built prism_trading-0.4.0.tar.gz and prism_trading-0.4.0-py3-none-any.whl
  PASS           reproducible
                 2 artifact(s) hash identically across two builds of the same tree
  PASS           artifact-identity
                 prism-trading 0.4.0, requires-python >=3.12, 6 core requirement(s) and 2 extra(s)
  PASS           contents
                 wheel 116 entries under prism, prism_trading-0.4.0.dist-info; sdist 414 entries;
                 no credential-shaped path in either
  PASS           fresh-install
                 the wheel installs into a fresh environment (13 distributions), imports,
                 reports engine_version 0.4.0, and its console script runs
  NOT EVALUATED  dependency-provenance
                 12 distribution(s) resolved for the core set: 6 declared and 6 transitive, each
                 with the sha256 of its installed RECORD; uv.lock does not describe prism-trading
                 0.4.0, so every dependency row read out of it is a row about a different build

  artifacts (sha256, bytes):
    a2b02711fb58e65d075e628b31c78787d3ff6fba0f37ab3a7117788f6045348c     817436  prism_trading-0.4.0-py3-none-any.whl
    277c1263c3c03eaaaea69ef620f9ad12ae5a1d955db9fa1a87d1c42e0e4f4c80    2072515  prism_trading-0.4.0.tar.gz
```

Four things worth reading off that run:

1. **The build is reproducible.** Two builds of the same tree hashed identically, so the hashes
   above identify the source and not the run. That is what makes an artifact hash useful to a
   third party: they can rebuild and compare rather than take the number on faith.
2. **The wheel is clean.** 116 entries, all under `prism/` and its `dist-info`. The sdist carries
   the repository, tests and docs included, at 414 entries.
3. **Six declared requirements become twelve installed distributions.** That is the transitive
   surface a lockfile pins, and section 2's point about scope in one number.
4. **The verdict is INCOMPLETE, not PASS, and the reason is the stale lockfile.** The check
   refuses to call a release verified while a dependency record it cannot trust is sitting in the
   repository. Nothing failed; something could not be established.
5. **The two hashes track different things.** Re-run at `git:5557eb5` after four documentation
   commits and one new script, none of which ship in the wheel: the wheel hash was unchanged at
   `a2b02711...` and only the sdist hash moved, since the sdist carries the repository (Run
   2026-09-09). The wheel hash names the installed artifact, including its code and
   metadata. In particular, README changes alter the package description inside
   the wheel without changing engine code. Expect the hashes here to differ from your
   own run whenever the tree differs; the source stamp printed above each set is what makes them
   comparable.

### What the check does not establish

- It does not audit any dependency's behaviour. See section 2.
- It does not verify a signature. There is none to verify: the artifacts here are unsigned, and
  signing is an owner act tied to a key this repository does not hold. The reproducible build plus
  the recorded hashes are what a third party can check today.
- It does not check the index-served artifact, because nothing is published. Once a release is
  published, the same check should be run against the downloaded artifact and its hash compared
  with the one recorded at build time.
- Its `fresh-install` check installs from a local file, so it exercises the artifact and not the
  index path.

Status: **Run** 2026-09-09, output above, exit 2 (verified separately from the pipeline, since a
pipe reports the exit code of the last command). `tests/test_release_check.py`: **Tested**,
36 passed.

### Check after the lock refresh

**Tested 2026-09-15**, at `git:f9a8a3a3a3d900ee8140d6e7690d58cef135b299`:
`python ops/release_check.py` passed all six checks, exit 0. Two builds produced
identical artifacts; the wheel installed into a fresh environment with thirteen
distributions including the engine; dependency provenance resolved twelve core
dependencies and verified the renamed project in the lock. The wheel contained
117 entries and the source distribution 420, with no credential-shaped paths.
This supersedes the historical INCOMPLETE verdict above for that source commit.
Subsequent documentation changes can change the wheel's README metadata and
the source archive; rerun the check for the exact artifact being installed.
Nothing was uploaded to an index.
