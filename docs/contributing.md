# Contributing: two review categories, and why merging code never arms an account

Date: 2026-09-09.

Prism separates two kinds of change that are usually conflated, because they answer to different
evidence and carry different risk. Every contribution is one or the other, and a contribution
that is both is split before it is reviewed.

- **An engine change** alters the machinery: what the code can express, how it computes, what it
  refuses. It is judged on correctness, on interface compatibility, and on tests.
- **A reference-policy change** alters what the reference strategy is: its signal parameters, its
  cost stack, its risk policy, its universe, its execution profile. It is judged on evidence,
  under the registration rules, and it is not an improvement because it improves a number.

The third rule sits over both: **merging code never authorizes a live configuration.** Section 3.

---

## 1. Engine changes

**What counts.** Anything under `src/prism`, which is the shipped wheel, plus the packaging that
decides which code an operator runs.

**What a reviewer asks.**

1. **Which compatibility surface does this touch, and does the version class allow it?**
   `docs/versioning.md` section 2 is the table. A removal is not a patch; a schema increment
   without a migration is not a minor.
2. **Can it change a decision?** Use the three-valued question from
   `docs/engine_overhaul_operating_changes.md` section 2: can it change the quantity, side, type
   or destination of an order (yes), can it stop a session from sending one (refuses), or neither
   (no). A change that is "yes" or "refuses" is named in the release notes whatever its size.
3. **Does a control refuse less than it did?** A refusal that becomes a warning is a substantive
   change and needs its own justification. The direction that needs no argument is a control
   refusing more.
4. **Does it hold the import boundary?** The production import path must not pull a research
   heavyweight into the process; `tests/test_import_hygiene.py` pins that closure by running the
   probe in a subprocess with a controlled `sys.path`. Adding a core dependency is a dependency
   decision, not an implementation detail.
5. **Is the evidence attached?** Every non-trivial claim in the change description carries its
   status: Tested (tests exist and pass), Run (executed, correctness not asserted), Drafted
   (written, not executed). "Should work" is not a status.
6. **Does it cross the code boundary?** `src/prism` is the wheel; `research/` imports it and never
   the reverse. Cross-boundary reuse is duplication by value with a citation, never an import
   (`SPEC.md` section 7).

**Tests.** Install the test environment with `uv sync --extra construction`; the
solver tests need that extra even though importing the engine does not.
`.venv/bin/python -m pytest -q -m "not research"` is the core suite. Run it from a separate
clone or worktree, never from a checkout that holds a deployment's `.env`: `tests/conftest.py`
sets `PRISM_DISABLE_DOTENV=1`, so no test process discovers a `.env`, but a separate checkout
keeps logs and caches out of the deployment too. A change to a
refusal path needs a test that the refusal happens, and a test that it happens *before* the side
effect it is protecting against, which is a different assertion.

---

## 2. Reference-policy changes

**What counts.** The reference configuration and everything that enters its identity:
`examples/reference/` (cost stacks, risk policies, execution profiles), `calibrations/`,
`registrations/`, `folds/`, the universe files under `data/universe/`, and the trial ledgers
under `trials/`.

**What a reviewer asks.**

1. **Is there a registration?** Register, then run (`SPEC.md` section 4). The registration is a
   tracked document under `docs/` following the `*_design.md` or `*_preregistration.md` pattern,
   or a declared registration file under `registrations/`, and the registration is the commit
   that adds it. A comparison that was not declared before it ran is exploratory and is recorded
   as exploratory. Standardization and weighting alternatives are not free diagnostics: each is a
   declared arm or it is not run.
2. **What does this change claim, and at what tier?** The tiers are `no_claim`,
   `mechanics_clean`, `gross_edge`, `net_edge`, `robust_edge`, and the record carries which one
   it reached.
3. **Does it promote anything?** Almost certainly not, and the burden is on the change to show
   otherwise. A same-sample comparison can kill a configuration and can never promote one; the
   reference book's promotion stays under the prospective rules of `SPEC.md` section 4, which
   require out-of-sample data after the registration date. That
   sentence is not editorial, it is what the stage 4 comparison record itself writes into its
   `promotion_check` field.
4. **Does the trial ledger stay honest?** Every evaluated configuration is a ledger row, NaN and
   degenerate outcomes included. Ledgers are append-only and editing or deleting a row is
   falsification. A claim deflates against its own selection set, never against a pooled ledger
   of incommensurable strategies and never against fewer rows than were run. Closed sets stay
   closed: a new idea is a new registration, not a new row in a set whose kill rule has fired
   (`SPEC.md` section 4).
5. **Does it rewrite a certified artifact?** Anything under `results/` cited by a file in
   `docs/certifications/` is never rewritten. A correction is a new artifact with a new id.
6. **Are the limitations updated?** If the change adds or moves a published number,
   `docs/limitations.md` moves with it in the same change. A result whose limitations are
   documented in a later change was published without them.

**An engine change that alters a reference number is still an engine change.** It is reviewed as
one, and the reference record is re-derived and re-recorded as a separate act, so that "the code
changed" and "the claim changed" never arrive as one undifferentiated diff.

---

## 3. Merging code never authorizes a live configuration

**The rule.** Approving and merging a change gives it no standing against any account. Whether a
given process may send an order to a given account is a separate decision, made by the account
owner, recorded locally, and enforced before any request. No review, no release, and no
maintainer can make that decision on an owner's behalf.

**The mechanism that implements it,** in `src/prism/live/authorization.py`. An authorization
record is what the account owner approved, and it binds seven fields: `mode`, `endpoint`,
`account_id`, `config_id`, `engine_version`, `execution_host` and `distribution_id`.
`AuthorizationGate.verify_binding` compares every one of them by exact string equality and raises
`AuthorizationRefused` on the first mismatch (`src/prism/live/authorization.py:868-871` for the
`config_id`, `engine_version` and `distribution_id` rows). The refusal happens before any
request: at the top of `loop.decide_and_submit`, ahead of the reconcile read, and again in
`AlpacaBroker.submit` before the request body is built. Nothing between the two constructs an
order request.

**Why that enforces the rule rather than merely restating it.**

- A merged engine change moves `engine_version`, or `distribution_id`, or both. The build stamp
  is `git:<commit>[+dirty]` in a checkout and `sha256:` of the installed `RECORD` from a wheel,
  so even a same-version rebuild of modified code is a different `distribution_id`
  (`src/prism/live/authorization.py::distribution_id`).
- A merged reference-policy change moves `config_id`, because `config_id` hashes the resolved
  specification plus the *content* of every policy, calibration and risk-model file it references,
  not their paths (`src/prism/spec.py::StrategySpec.identity_payload`). Editing a cost stack in
  place changes the identity.
- `config_id` also includes `engine_version` (`src/prism/spec.py:339`), so an engine change moves
  the configuration identity too. Both routes lead to the same place.
- Absence is not permission. A run directory with no authorization record is operating state S4b,
  which permits nothing at all, every broker write included
  (`src/prism/live/authorization.py::AuthorizationGate.verify_binding`, the no-record branch).
- An entry point that resolves no specification computes no `config_id`, and therefore cannot
  bind. It is refused with an explanation rather than allowed through on a partial match
  (`src/prism/live/authorization.py::refuse_an_unbindable_record`).

Tested: `tests/test_authorization.py` refuses a live order with no authorization record, with a
mismatched account id, `config_id`, `engine_version`, `distribution_id` or execution host, with an
expired record, and with a live endpoint under paper mode. Each refusal is proved by a broker
double that raises on any call, so the assertion is that the order request was never constructed,
which is a stronger fact than the venue having rejected it. The test carrying this section's rule
in its own docstring is `test_live_order_refused_on_a_mismatched_config_id`: "Merging code never
authorizes a live configuration".

**What the mechanism is not.** Local authorization records what the account owner approved. It is
not a credential, not a signature, not a profitability certificate, and not a sandbox against
arbitrary private Python. Its integrity rests on filesystem permissions. It answers exactly one
question: did the owner approve this process, on this host, against this account, for this
action.

**The owner acts it cannot replace.** Opening, funding or resizing a real-money account, arming
crash-conditional de-grossing, changing the live book's configuration, restarting or repointing
the scheduler, and publishing a distribution. None of these is a merge.

---

## 4. Practical rules for a contribution

- **Never contribute a credential.** Not in a test fixture, not in a recorded artifact, not in a
  log excerpt. `ops/release_check.py` refuses to bless a distribution carrying a credential-shaped
  path, but the check is a backstop and not a review.
- **Never contribute private data.** Private code, trials, fills and datasets stay local by
  default. An opt-in path for narrowly described negative results and appropriately aggregated
  execution observations is intended, and until it exists the answer is no. Anonymization is not
  automatically sufficient, and data rights are the contributor's to establish, not the project's
  to assume.
- **Preserve identifiers.** Old research ids, run ids, fold ids and citations are never renamed.
  A cited artifact that no longer exists is recorded as missing rather than silently dropped.
- **Format only what you touched.** isort with the black profile on touched files; never repo-wide.
- **Commit style** `type(scope): Sentence-case summary`, no attribution trailers.
- **Do not cite `docs/dev/`.** It is gitignored scratch and a tracked file that cites it is broken
  for everyone else.
