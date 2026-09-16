# Versions, compatibility, deprecation, and what to re-verify after an upgrade

Date: 2026-09-09. Applies to the distribution `prism-trading`, import name `prism`,
at version 0.4.0 and after.

This document is the promise attached to a version number, and, just as important, the list of
things a version number does not promise. Read section 1 before section 2: the compatibility
table is only meaningful once you know which class of change it governs.

The engine's own operating changes for the current overhaul are listed in
`docs/engine_overhaul_operating_changes.md` and are not repeated here. That document says what
changed and why; this one says what a user may infer from the version number attached to it.

---

## 1. What a version number promises, and what it does not

`MAJOR.MINOR.PATCH`. The installed value is read by `prism.spec.engine_version()`
(`src/prism/spec.py:927`) from `importlib.metadata.version("prism-trading")`, falling back to
`prism.__version__` in a source tree with nothing installed.

**The version governs interfaces, not decisions.** Four surfaces are under the compatibility
contract:

1. The public Python interface: the classes and functions a private strategy project imports to
   declare a component, a specification, a cost stack, a risk policy, or a broker.
2. The command surface: `prism <command>` and its arguments.
3. The on-disk formats: the specification schema, the durable loop state
   (`STATE_SCHEMA_VERSION`, `src/prism/live/state.py:66`), the authorization record schema, the
   result and trial record schemas, and the claim packet.
4. The declared behaviour of the refusal controls: authorization, the account-keyed execution
   lock, freshness, and the endpoint allowlist. A control that refuses today does not start
   permitting under a patch.

**Decisions are outside that contract.** A release can change what the engine decides from the
same specification and the same data. It happens for ordinary reasons: a defect fix in ranking,
in rounding, in the cost stack, in the calendar. There is no version increment that means "your
book will not move", because the alternative is refusing to fix defects. What the project
promises instead is that a decision change is *visible before it is adopted*, which is section 3.

Three further things the number does not promise:

- **Not identity.** `config_id` includes `engine_version` in the payload it hashes
  (`src/prism/spec.py:339`), so the same specification file resolves to a different `config_id`
  under a different engine. Results recorded under two engine versions are not the same
  configuration, and the record says so rather than papering over it.
- **Not the reference strategy.** An engine release does not change or promote the reference
  configuration. The reference book is versioned separately (launch brief Q2,
  `docs/prism_engine_decisions.md`), and its promotion stays under the prospective rules of
  `SPEC.md` section 4.
- **Not profitability, and not safety.** Local authorization records what the account owner
  approved. It is not a profitability certificate and does not sandbox private Python
  (`src/prism/live/authorization.py`, module docstring).

**0.x specifically.** While MAJOR is 0, MINOR carries the breaking-change role that MAJOR carries
after 1.0. Do not read `0.4 -> 0.5` as a compatible step because the leading digit is unchanged.

---

## 2. What may change in each release class

| | Patch (0.4.0 -> 0.4.1) | Minor (0.4 -> 0.5) | Major (0.x -> 1.0) |
|---|---|---|---|
| Public Python interface | Nothing removed or renamed. Additions allowed. | Additions and deprecations. Nothing removed. | Removals of interfaces deprecated for at least one prior minor. |
| Command surface | No command or argument removed; no default changed. | New commands and optional arguments; a default may change only if the release notes name it. | A command may be removed or renamed. |
| On-disk schema | No version increment. | May increment, and the read path migrates the old version automatically, as the v2 to v3 loop-state step does. | May increment without a backwards read; the migration is then a documented one-time step. |
| Refusal controls | May refuse strictly more, never strictly less. | May refuse more; a relaxation is named in the release notes with its justification. | Same, with the control set itself open to change. |
| Dependencies | A floor may rise only if the existing floor is unusable. A dependency is never added. | Dependencies may be added, moved between extras, or dropped. | Same. |
| Decisions | May change, only through a defect fix, and the release notes name the fix and the affected path. | May change. | May change. |

Two rules cut across the table.

**A decision change is never silent.** Every release whose changes can reach the decision path
says so in its release notes, names the mechanism, and the change is reviewable with the
comparison in section 3. "Can reach the decision path" is judged the way
`docs/engine_overhaul_operating_changes.md` section 2 judges it: does the change alter the
quantity, side, type or destination of an order, can it stop a session from sending one, or
neither.

**A refusal is not a regression.** If an upgrade starts refusing something the previous version
permitted, that is a control doing its job, not a compatibility break. It is documented as a
behaviour change with its reason, and it is not grounds for a patch that restores the old
permissiveness.

---

## 3. The upgrade comparison, and what it can and cannot compare

The engine ships a decision-level upgrade comparison so an upgrade can be reviewed instead of
adopted on faith, and so nobody has to keep a fork alive beside the live one.

**What ships today, precisely.** `prism upgrade-diff` and the equivalent
`prism replay --upgrade-diff` are implemented in `src/prism/cli.py` and call
`prism.live.replay.upgrade_diff_report`. They compare two previously produced
captures. They do not install another engine version or produce its candidate
capture inside the current process.

```sh
prism upgrade-diff --sessions /absolute/path/to/baseline-capture \
  --candidate /absolute/path/to/candidate-capture --json upgrade-review.json
prism replay /absolute/path/to/baseline-capture --upgrade-diff \
  --candidate /absolute/path/to/candidate-capture --json replay-review.json
```

`--sessions` names the requested session capture and also supplies the default
baseline; `--baseline` selects a separate incumbent capture when needed. Repeat
`--bar YYYY-MM-DD` to restrict the review. The default weight and share tolerances
are `1e-9`, configurable through `--weight-tolerance` and `--share-tolerance`.
`--json` writes the complete report beside the printed summary.

Exit **0** means every requested session was comparable and its compared decision
fields agreed; exit **1** means at least one compared decision changed or the
invocation failed; exit **2** means nothing failed but at least one requested
session could not be compared. Inspect `orders_compared` separately: agreement
on target weights does not establish order agreement when order records are absent.
A missing candidate produces an incomplete report, never an unchanged-upgrade verdict.
Plain `prism replay` without `--upgrade-diff` refuses and names
`python -m prism.scripts.replay_loop --help` for the separate replay driver.

**Status: Tested.** The installed-package onboarding check asserts all three
upgrade-review outcomes on disposable synthetic captures: a missing candidate,
equal targets, and a changed target. Its [receipt](onboarding_verification.json)
and [walkthrough](onboarding.md) state the input and evidence limits. The existing
`tests/test_cli.py` and `tests/test_upgrade_diff.py` cover the CLI and comparison
mechanism; this synthetic receipt does not prove parity for the paper capture.

The Python interfaces remain available: `upgrade_diff`, `upgrade_diff_report`,
`recorded_decisions`, and `replayed_decisions` in `prism.live.replay`.
`python -m prism.live.parity_gate` drives the broader recorded-session gate.

**What it compares.** Decisions, and only decisions: the target book each engine version
constructed for each recorded session, and the signed share orders that book produced, per
session, naming which securities enter, which leave, which move and by how much, at a declared
weight and share tolerance (default 1e-9 each). It reports whether each version re-decided at
all, because two engines that both held the cadence have not disagreed.

**What it refuses to compare.** Outcomes. Fill prices, realized equity and execution quality
belong to the venue and the day, not to the engine version, so a difference in them is not
evidence about an upgrade (`src/prism/live/replay.py`, module docstring). Order quantities are
compared only when both sides recorded an order set; a side that recorded none leaves
`orders_compared` false rather than reporting that every order changed.

**What it needs, and what the shipped fixture does not have.** The candidate side has to be
produced, which means running the upgraded engine over the recorded sessions, which needs the bar
panels those sessions scored. The shipped capture has none. Its manifest records
`panels.status = "unavailable"`, 119 of 119 symbols with no bar at any of the five sessions, and
the stage 5 parity gate reports decision-parity NOT COMPARED, 0 of 5 sessions
(`docs/engine_overhaul_parity_gate.json`; the reasoning is `docs/engine_overhaul_operating_changes.md`
section 1.3). Run 2026-09-09 against the shipped fixture with no candidate supplied:

```
upgrade-diff: baseline (git:667bb09) -> candidate (unknown)
0/5 session(s) compared, 0 with a different decision
  2026-07-13: NOT COMPARED - candidate (unknown): no candidate capture was supplied ...
  ...
This report does not say the upgrade changes nothing: it says nothing at all about the sessions
above.
```

That last line is the design. An empty difference set over sessions nobody computed is not
evidence of an unchanged engine, and the report refuses to be read that way.

To get a real comparison, supply what `docs/engine_overhaul_operating_changes.md` section 1.3
enumerates: a bar source covering the sessions offline, the full scored symbol set rather than
the traded subset (a decile boundary is a property of the whole scored panel), the dates
including the signal's trailing history, and the adjustment basis the vendor served on those
dates. A panel rebuilt today may sit on a different split-adjustment basis than the session did,
which makes the comparison a bounded claim about the engine's arithmetic rather than a
reproduction of the session.

**The honest summary.** The comparison is the right instrument and it is real, tested code. What
it is not, today, is a turnkey report: it requires a bar panel the engine does not currently
persist, and until a user supplies one it will correctly tell them it compared nothing.

---

## 4. Deprecation

**How long a deprecated interface keeps working.** At least one full minor release after the
release that deprecates it, and it is removed only in a major release. In practice that means: an
interface deprecated in 0.4 keeps working through 0.5 at minimum, and disappears no earlier than
1.0. A deprecation that would have to move faster than that is a breaking change and takes the
breaking-change path instead.

**How it announces itself.** Three ways, chosen by what the interface is, all already in use:

- A console entry point kept as an alias prints one line naming its replacement, before it does
  any work, on stderr. `prism.scripts.announce_alias`
  (`src/prism/scripts/__init__.py:18`) is the mechanism, wired at `prism-doctor`,
  `prism-build-universe` and `python -m prism.scripts.paper_loop` (commit `56e7405`). Stderr and
  not stdout, so a consumer parsing the command's output sees nothing new.
- A Python attribute or module emits `DeprecationWarning` at the point of use, naming the
  replacement. `prism.config.__getattr__` does this for the legacy directory constants.
- A function kept only to read recorded artifacts carries a module-level note saying what it is
  for and when it goes. `prism.validation.metrics.LEGACY_DIAGNOSTIC_NOTE`
  (`src/prism/validation/metrics.py:554`) is the pattern.

**What the announcement must contain.** The name of the deprecated thing, the name of its
replacement, and the release class that will remove it. An announcement that says only
"deprecated" is not one.

**What is exempt.** Nothing in the public interface. Private names (a leading underscore, or a
module the documentation does not reference) may change in any release without notice, which is
what makes the public surface small enough to keep a promise about.

**Research identifiers are never deprecated.** Old research ids, run ids, fold table ids and
citations are preserved so recorded evidence stays readable and re-derivable. A renamed interface
does not rename the history that used it.

---

## 5. What to re-verify after an upgrade

In this order. Steps 1 to 5 are offline and cost minutes. Step 6 is an owner act.

**1. Confirm what you are actually running.** Two identities, not one.
`prism.spec.engine_version()` gives the distribution version;
`prism.live.authorization.distribution_id()` gives the build, which is `git:<commit>[+dirty]` in
a checkout and `sha256:` of the installed `RECORD` from a wheel. A version string alone cannot
distinguish a pinned release from a modified checkout carrying the same string, which is why the
authorization record binds both.

**2. Run the offline suite.** `.venv/bin/python -m pytest -q` in the checkout, or the private
project's own smoke workflow against the installed wheel.

**3. Re-resolve your configuration and compare `config_id`.** It will differ, because
`engine_version` is inside the hash (`src/prism/spec.py:339`). That is expected. What matters is
that resolution still succeeds and that no policy file, calibration file or risk model failed to
resolve, since those enter the identity as content.

**4. Run the upgrade comparison over recorded sessions.** Section 3. Adopt the upgrade only when
every difference it reports is explained, and read a NOT COMPARED verdict as an absence of
evidence rather than as agreement.

**5. Re-run your evaluation on the frozen fold table and compare with the recorded result.** A
difference is not automatically a defect, but it is not automatically fine either: it must be
attributable to a change the release notes name. The fold table is an explicit frozen table of
train and test date ranges, content-hashed and recorded in the result as `fold_geometry`
(`results/demotion_b1/folds.json` is the recorded one), precisely so this comparison is
possible.

**6. Re-grant authorization.** This is the step that cannot be skipped, and the mechanism makes
sure of it: the authorization record binds `config_id`, `engine_version` and `distribution_id`,
and `AuthorizationGate.verify_binding` compares each by exact string equality and refuses before
any broker request when one differs (`src/prism/live/authorization.py:868-871`). An upgrade
therefore moves at least two of those three fields, every existing grant stops matching, and the
first live session after an upgrade refuses until the owner writes a new record. Upgrading is not
a way to change what a live account is doing. (Tested: `tests/test_authorization.py` refuses a live order on
a mismatched `config_id`, a mismatched `engine_version` and a mismatched `distribution_id`,
in each case before any request reaches the broker.)

**7. Check the durable state migrated.** If the loop-state schema incremented, the migration runs
in the read path: `LoopState.load` forwards the payload to `migrate_state_payload`
(`src/prism/live/state.py:375-379`), and a payload that lands on an unsupported version raises
rather than starting flat (`src/prism/live/state.py:613-617`: "migrate explicitly, do not start
flat"). A state file that cannot migrate is a refusal, never a silent reset.

---

## 6. Release notes

Every release carries notes with, at minimum: the version, the four compatibility surfaces of
section 1 and what moved in each, whether the release can change decisions and through which
mechanism, the deprecations introduced and the release class that will remove them, and the
artifact hashes from `ops/release_check.py`. See `docs/depending_on_prism.md` for the check and
for how to pin a release once it exists.

Status of this document: the policy in sections 1, 2, 4 and 6 is **Drafted**, meaning it is the
rule from this release forward and no prior release was made under it. The mechanisms cited in
sections 3 and 5 are **Run** or **Tested** as marked at each claim.
