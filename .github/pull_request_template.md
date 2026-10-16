## What this changes

<!-- One or two sentences, and the issue it addresses if there is one. -->

## Kind of change

<!-- docs/contributing.md explains the difference. A change that is both is split in two. -->

- [ ] Engine change: code under `src/prism`, or the packaging
- [ ] Reference-policy change: the reference strategy's parameters, cost stack, risk policy,
      universe, calibrations, registrations or trial logs
- [ ] Documentation, tests or tooling only

If this can change an order's quantity, side, type or destination, or stop a session from sending
an order, say so here:

## Evidence

<!-- Mark each claim above as Tested (tests exist and pass), Run (executed, correctness not
checked) or Drafted (written, not executed). -->

- [ ] The core suite passes: `uv sync --extra construction`, then
      `.venv/bin/python -m pytest -q -m "not research"`
- [ ] isort (black profile) was run on the touched files only, not repo-wide
- [ ] A new or changed refusal has a test that it refuses, and that it refuses before the action
      it guards

## Records

- [ ] No file under `results/` that `docs/certifications/` cites is rewritten
- [ ] Trial logs under `trials/` are only appended to; no row is edited or removed
- [ ] A new experiment's pre-registration was committed before any of its configurations was
      evaluated (`SPEC.md` section 4)
- [ ] A published number that is added or changed is updated in `docs/limitations.md` in the
      same change

## Safety

- [ ] The diff, fixtures and pasted logs contain no credentials, account numbers, alert or
      heartbeat URLs, or private data
- [ ] Commit messages follow `type(scope): Sentence-case summary`
