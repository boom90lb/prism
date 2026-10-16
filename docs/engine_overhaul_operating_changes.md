# Engine overhaul: intentional operating changes, the parity gate, and the observation window

Date: 2026-09-09. Branch `engine-overhaul`, 33 commits from `c781fb8` (stage 0) to `9c9c765`
(stage 5). This document is the thing the engine specification's stage 5 paragraph asks for
before the switch: "every intentional difference is explained and approved under the existing
operating rules before the switch; only then is the live checkout advanced and the wrappers
repointed at the pinned release; the declared paper observation window then runs."

Three parts, in the order an owner needs them:

1. What the parity gate actually established, and what it could not.
2. Every intentional behaviour change to the operating path, with its stage, its justification,
   whether it changes what the engine does with an order, and whether it is already approved.
3. The observation window: what is observed, for how long, what passes, what stops.

Read part 1 first. It bounds what parts 2 and 3 can be judged against.

---

## 1. The parity gate

### 1.1 How to run it

```
python -m prism.live.parity_gate \
    --capture tests/fixtures/paper_replay \
    --store-dir data/store \
    --out docs/engine_overhaul_parity_gate.json
```

Exit 0 every check passed, 1 a check failed, 2 a check could not be evaluated. The declared
tolerance is written into the artifact, not into this document, so a report can never drift from
the numbers it was judged against. The module is `src/prism/live/parity_gate.py`; its tests are
`tests/test_parity_gate.py` (17 tests, passing).

### 1.2 What it reported on 2026-09-09

Verdict INCOMPLETE, exit 2. Artifact: `docs/engine_overhaul_parity_gate.json`.

| Check | Verdict | What it says |
|---|---|---|
| capture-integrity | PASS | 8 recorded ledgers hash to the digests the capture wrote |
| recorded-extraction | PASS | 5 of 5 sessions recovered; per-session target counts equal the counts `capture.py` recorded independently |
| extraction-parity | PASS | 5 of 5 sessions compared, 0 differing at the declared tolerance, orders compared on both decision sessions |
| order-sizing-parity | PASS | 209 recorded parent orders over 2 decision sessions rebuilt from the recorded decision inputs, 0 differing at a zero share tolerance |
| refutation | PASS | a 10x-tolerance weight move and a one-share order change are both reported as differences |
| decision-parity | **NOT COMPARED** | 0 of 5 sessions can be compared end to end |

The declared tolerance, as the artifact carries it: weight 1e-9 absolute per name, which is
$0.000928 at the capture's smallest recorded equity of 927,895.565 and 6.85e-5 shares of the
capture's cheapest recorded reference price of 13.55; share tolerance 0.0, because orders leave
this engine as whole shares and there is no rounding band that makes two order sets the same.

### 1.3 What parity is demonstrable, and what it is not

**Demonstrable, and demonstrated.** For each of the two decision-bearing sessions the gate
rebuilds the parent order set from the recorded decision inputs (the decided book, the marked
equity and the decision reference prices from `targets.jsonl`, and a position book carried
forward from `fills.jsonl` by settle date) through `prism.assembly.size_orders`, which is the one
sizing implementation the live protocol and the evaluator now share
(`src/prism/live/loop.py:206` `targets_to_orders` delegates to it). All 209 orders match the
record share for share: 98 of 98 at 2026-07-13 from a flat opening book, 111 of 111 at 2026-08-11
from the 98-position book the first session's fills produced. That exercises settle, mark and
order arithmetic against the recorded ledgers, including the seven orders the whole-share and
dust rules censored at the second session.

Two inputs are named rather than assumed. The sizing knobs are the shipped momentum book's
(`min_order_notional` 1.0, `whole_shares` True, `order_id_prefix` "mom:";
`src/prism/scripts/paper_books.py:77-78`, `:215`), which the capture's own client order ids
confirm. And the 20 names the second session exited carry no price in the targets ledger, because
that ledger prices the book and not the exits, so those take their decision price from the order
ledger's own `reference_price`. The gate reports that count per session rather than hiding it. It
is a weaker input than the rest.

The same claim, in a stronger form that also runs the composition, band and participation stages
and reproduces the recorded equity mark, is `tests/test_assembly_replay_parity.py`, landed at
stage 1 (`29a6c67`). The gate does not invent this parity; it aggregates it into one artifact
with a declared tolerance and a verdict.

**Not demonstrable here.** The decision itself. The capture's manifest records
`panels.status = "unavailable"`: 119 of 119 symbols have no bar at any of the five sessions. The
gate probed this again today rather than quoting it. Against the live checkout's `data/store`,
2 of 119 symbols have any cache at all, the newest bar anywhere in the store is 2026-07-06, and
0 of 119 have a bar at any of the five session dates. So the candidate side of a decision
comparison cannot be produced, 0 of 5 sessions are compared end to end, and the gate reports NOT
COMPARED with the reason. It is never reported as a pass.

**What would close it.** Four things, all inputs, none of them code:

- **Which bar source.** Any source that can serve daily open and close bars for these dates
  offline. The reason none exists is mechanical: with `--bar-source alpaca`, which is the default
  and what `ops/paper_loop_nightly.sh:127-132` runs, the loop constructs
  `AlpacaBarSource.from_env()` (`src/prism/scripts/paper_loop.py:337-338`), and that class takes
  no cache directory and no store (`src/prism/live/alpaca_data.py:78-91`). Nothing it fetches is
  written anywhere. The legacy Twelve Data caches under the live checkout's `data/` do cover all 119
  symbols but stop at 2026-07-07, which is before the first captured session.
- **Which symbols.** The roughly 503 the sessions actually scored, not the 119 the capture
  trimmed to. A decile boundary is a property of the whole scored panel, so the trimmed set
  cannot re-derive the ranking. `data/universe/sp500_pit_resolved_2026-07-17.txt` is the closest
  point-in-time membership list in the tree.
- **Which dates.** 2025-06-09 through 2026-08-14 inclusive: the 274 trailing sessions of history
  the signal requires before the first decision bar 2026-07-13 (252 lookback plus 21 skip plus 1,
  which the manifest records as `min_history_required`), plus the 25 sessions from 2026-07-13 to
  the last captured session.
- **Which adjustment basis.** The one the vendor served on those dates. A panel rebuilt today may
  sit on a different basis after a back-rewrite, which is why `src/prism/io/store.py:50` carries
  `SplitRewriteDetected` at all. A rebuilt panel therefore makes replay parity a bounded claim
  about the engine's arithmetic and not an unqualified reproduction of the session.

There is a fifth gap the capture cannot close at any price: the specification asks parity to be
judged on "at least three decision-bearing sessions" and records that this is "not cuttable from
the record" (spec line 5514), because `targets.jsonl` holds exactly two rows.
A third refresh has to come from a session the loop actually runs. That is what the observation
window in part 3 is sized to supply.

---

## 2. The intentional differences

Scope: the operating path, meaning everything that runs when the nightly session runs, plus the
commands run against a paper or live account (`run`, `check`, `doctor`, `reconcile`,
`upgrade-diff`), plus the wrappers and the packaging that decide which code that is. Research and
evidence-path changes are listed separately in 2.3 so nothing is silently dropped.

"Changes an order?" has three values, because an owner needs the distinction:

- **Yes**: can change the quantity, side, type or destination of an order the engine sends.
- **Refuses**: cannot change an order's content, but can stop a session from sending any.
- **No**: cannot do either.

Count: 24 intentional differences on the operating path. 15 of them change order behaviour
(7 "Yes", 8 "Refuses") and 9 do not.

### 2.1 Order-affecting

| # | Change | Stage | Order? | Approved? |
|---|---|---|---|---|
| O1 | The endpoint check becomes a refusal, not a warning | 5 | Refuses | Yes |
| O2 | The execution lock loses warn-and-proceed | 5 | Refuses | Yes |
| O3 | Authorization refuses before the request reaches the broker | 5 | Refuses | Yes |
| O4 | An expired grant narrows scope instead of voiding the record | 5 | Yes | Yes |
| O5 | Restart recovery asks the venue for each order's fate | 5 | Yes | Yes |
| O6 | A coverage shortfall blocks new exposure per sid | 5 | Yes | Yes |
| O7 | A halted name is carried stale and blocks new exposure in that sid | 5 | Yes | Yes |
| O8 | A held name leaving eligibility is pinned reduce-only, not exited | 3 (spec), landed 4 | Yes | Correction yes, **live wiring no** |
| O9 | A NaN holding is refused, not read as flat | 4 | Refuses | Yes |
| O10 | Order type and time in force come off the profile's `order_type_map` | 3b | Yes | Yes |
| O11 | The capabilities gate revalidates the batch before submission | 3b | Refuses | Yes |
| O12 | `--account paper\|live` is mandatory with no default | 5 | Refuses | Yes |
| O13 | Durable state moves to schema 3 with an explicit migration and a hard stop | 3b | Refuses | Yes |
| O14 | The ops wrappers take a release, and an unresolvable one refuses the session | 5 | Yes | Mechanism yes, **the flip is the owner's act** |
| O15 | The four risk-profile ids become files, tighten-only enforced at load | 3b | Refuses | Yes |

**O1. The endpoint check becomes a refusal.** `2aa38c5`. Two halves. The adapter enforces a
per-mode endpoint allowlist by exact string equality where the client is constructed, so
`AlpacaBroker.from_env(base_url=LIVE_BASE_URL)` under the default paper mode is refused however
the endpoint arrived, and `mode="live"` is how real money is selected. And `scripts/doctor.py`'s
`alpaca-endpoint` row is a FAIL rather than a WARN, whose docstring names it "a FAIL, not a
WARN" (`src/prism/scripts/doctor.py:151-164`), and that matters because a WARN does not change
the exit code the nightly alert reads. Justification: draft 1.1's rule was "the environment may not name the
live endpoint, but an explicit argument may", and section 3.10 replaces it with the allowlist.
Order effect: a session that would previously have sent orders to the live endpoint under paper
mode now sends none. Approved by the specification's stage 5 paragraph, which names
"`scripts/doctor.py`'s endpoint warning becoming a refusal under the wrong mode" explicitly.

**O2. The execution lock loses warn-and-proceed, and is now actually taken.** `e225bcc` and
`9c9c765`, with the section 7 migration row for "One checkout, one writer at a time"
(`SPEC.md:155`; `live/lockfile.py:47` warned and proceeded without `fcntl`). The lock is
account-keyed, created with `O_EXCL` in a per-host lock directory on the declared execution host,
holding process id, host and heartbeat; it is taken by every mutating entry point before its first
state load and released after its last state save, re-entrantly so a driver may wrap a whole
cycle. `src/prism/live/lockfile.py:22-23` states the rule: the `O_EXCL` creation is the lock,
there is no warn-and-proceed. The run-directory flock is retained as a second, weaker guard.

`9c9c765` closes a gap worth knowing about: until it landed, the account lock was taken only when
a driver passed `execution_lock=`, and no driver did, so no invocation held one. The lock is now
derived from the authorization record, which is already the single declaration of which account
runs from which host, and it arms only where the record names this host. An explicit spec still
wins as the operator's override. **The consequence for the window: a context with no
authorization record takes no account lock and keeps only the run-directory lock it has today.**
Order effect: a second concurrent writer on the same account submits nothing instead of proceeding
with a warning. Approved (Q5, Q17).

**O3. Authorization refuses before the request.** `2aa38c5`, `src/prism/live/authorization.py`.
Seven refusals: no record, a mismatched account id, `config_id`, `engine_version` or execution
host, an expired record, and a live endpoint under paper mode. Each is proved by a broker double
that fails the test on any call, so the test asserts the order request was never constructed
rather than that an exception was raised somewhere. `decide_and_submit` verifies the record above
its state load and above the reconcile read; the sized batch is classified and authorized as a
unit at write-ahead time, before the persist and before the first submit, because dropping one
side of a balanced batch can leave the portfolio outside its limits. Order effect: a process the
record does not bind sends no orders at all. Approved (section 3.10). **Backward compatibility
worth knowing:** under `--account paper` an absent record is the pre-authorization paper
instrument and passes with a note (`src/prism/scripts/doctor.py:186-198`); under `--account live`
an absent record is state S4b, which permits nothing, C0 included. So the paper loop keeps
running today without a record, which is also why the observation window in part 3 asks for one
to be written.

**O4. An expired grant narrows scope instead of voiding the record.** `2aa38c5`. Expiry moves the
grant to S4a: `add_exposure` goes, `manage_existing` stays, so C0 (cancel an open order) and C1
(reduce toward zero) survive an expired grant, and only S4b (no record at all) removes C0 as
well. Justification: the alternative reading, that expiry voids the record, would leave a book
that cannot be cancelled or reduced, which is the worst state to be stuck in. Order effect, both
directions: under an expired grant the engine still cancels and reduces where an all-or-nothing
reading would have stopped everything, and it opens nothing where a permissive reading would have
opened. Approved (section 3.10's action table).

**O5. Restart recovery asks the venue for each persisted order's fate.** `437d257`. The
write-ahead record says what was decided, not what reached the venue, so a resumed decision now
asks the venue for each persisted order's fate before retrying anything: working, terminal, never
received, or unknown when the venue will not say. Unknown is operating state S3, which permits
cancellation and reconciliation only and no retry until the fate is known. "No open orders" and
"cannot list open orders" stay different facts: an adapter that cannot answer puts its orders in
the unknown class rather than being read as a venue holding none. Order effect: a resumed
decision no longer retries an order whose fate is unknown, and cancels what the venue will not
account for. Exactly-once delivery is not promised; the accounting is idempotent, so a cancelled
order that turns out to have executed still settles into the same ledger row. Approved (section
3.10, and the stage 5 test list's "restart mid-session").

**O6. A coverage shortfall blocks new exposure per sid.** `183a6bd`. S1 is defined per sid, not
account-wide, and emitting it account-wide was a bug the tests found: an account-level S1
intersects into every name's permitted set, so one vendor missing one symbol would stop the other
four hundred and ninety-nine. The covered name keeps C3; the uncovered name loses C3 and keeps
C0, C1 and OB; `authorize_batch` refuses the batch naming the uncovered sid alone. Order effect: a
one-symbol data outage stops new exposure in that symbol and nothing else, where previously
nothing stopped. Approved (sections 3.10 and 3.11).

**O7. A halted name is carried at a stale mark.** `183a6bd`, with the reading of N7 that section 7
invariant 2 fixes. Halt status reaches freshness through the events table, not the session table,
which describes regular sessions and deliberately not intraday halts. A security under an open
halt is carried with a stale mark, because a halted name has no trade to mark against and its
last close is not a current price. The consequence is not new policy: the cancel and the reduction
all the way to flat survive `permitted_classes` unchanged; the partial reduction and the new
exposure do not. A stale valuation never simulates a sale. Order effect: no order opens in a
halted name. Approved (section 7, invariant conflict 2).

**O8. A held name leaving eligibility is pinned reduce-only rather than exited.** The
specification calls this a stage 3 correction (spec lines 2853-2863) with `eligibility_exit` in
`[construction]` as the switch, `reduce_only` the reference value from stage 3 and `exit` retained
for reproducing recorded sessions "so the stage 5 replay parity gate runs the recorded behavior
and the correction is not silently inside the parity number". **Where it actually landed:** on the
evaluator's decision path (`src/prism/decisions.py:573-577`, which fills a held name's absent
target from `ctx.w0` rather than forcing it flat) and in the convex constructor
(`src/prism/portfolio/convex.py:340-347`, which accepts only `reduce_only`). **Where it has not
landed:** the live decile book. `src/prism/portfolio/construct.py:332-334` still documents and
implements "the book fully exits a name that leaves the decile", and
`git diff c781fb8..HEAD -- src/prism/portfolio/construct.py` is empty, so nothing on this branch
changed it. Order effect when wired: a name leaving the decile is no longer forced flat on a bar
where the exit may not be executable. **Approval status: the correction is approved by the
specification (Q15); wiring it into the live decile book is not done and needs the owner's
decision before it is, because it changes an order the engine would otherwise send.** Note that
this is also the one change whose absence keeps the parity gate honest: with the live path still
on `exit`, the gate's arithmetic reproduces the recorded behaviour rather than the corrected one.

**O9. A NaN holding is refused rather than read as flat.** `5f9ff9d`. `decide_bar` filled its `w0`
and `positions` vectors with `fillna(0.0)` after reindexing, so a name the caller stated as NaN
was read as "not held", which tells the constructor the account is flat in a name it may be short
and turns a doubling into an "open". `_aligned_holding` now separates the cases: a sid the caller
never mentions is not held and carries 0.0, a sid it states as non-finite raises, a duplicated sid
raises. The path was unreachable because the evaluator's own call site refuses an unpriced held
name first, which is why it needed a test of its own. Order effect: refuses rather than sizing
against a wrong `w0`. Approved (N7).

**O10. Order type and time in force come off the profile.** `44d8812`. They now come off the order
through the profile's `[order_type_map]` when it carries one, and fall back to the constructed
default when it does not, which is what every current caller passes, so the paper loop and the
morning sweep are unchanged today. The OPG whole-share guard stays and still raises before any
request is made. Order effect: none today; the mechanism can change an order's type the day a
profile declares one. Approved (section 3.9).

**O11. The capabilities gate revalidates the batch before submission.** `44d8812`,
`src/prism/live/capabilities.py`. Rows refreshed or proven fresh against `[capabilities]
max_age_hours`, buying power read from the venue and compared against what the batch consumes, the
account checked against the one the batch was decided for, and the whole batch revalidated with
`require_capabilities = True`. It returns a report and decides nothing; the refusal is the
caller's. A capability row states only what the venue reported, and an unrecognised asset
attribute becomes a restriction recorded by name, which `limits.py` reads as blocking new
exposure. Order effect: a batch that exceeds buying power or names a restricted security is
refused before submission. Approved (sections 3.2 and 3.9).

**O12. `--account paper|live` is mandatory with no default.** Stage 5, pinned by
`tests/test_cli.py:169-174`. `prism run` has no default because it may write to a venue; `prism
check` takes a `paper` default because it writes nothing and reaches no venue. Order effect: a run
that does not name its account exits 2 and submits nothing. Approved (Q5).

**O13. Durable state moves to schema 3.** D4, and the section 7 migration row for
`STATE_SCHEMA_VERSION`. `af35dc8` implements it: positions become unmarked (stale) `Positions`,
cash and the three bar markers are copied, `fills_marker` starts from `last_settled_bar`, and
pending orders keep their client order ids verbatim, because regenerating one would resubmit a
live order. The v1 to v2 step is kept ahead of it and any other version is a hard stop that
refuses to start rather than starting flat (`src/prism/live/state.py:613-617`). The v2 vocabulary
survives as views onto the portfolio rather than as a second copy. Order effect: none while the
migration succeeds, and the live paper book (99 positions, 937,669.47 in cash) is the acceptance
test, copied out of the run directory, migrated, and compared field by field including equity at
the prices that book recorded. A document that fails to migrate refuses the session. Approved
(D4, section 7 row).

**O14. The ops wrappers take a release.** `437d257`. `PRISM_RELEASE` defaults to `checkout`, which
is exactly what runs today, so upgrading the wrapper alone changes nothing about the session that
trades. A release that does not resolve to an executable interpreter refuses the session and
alerts rather than falling back to the checkout
(`ops/paper_loop_nightly.sh:96-110`), and the provenance stamp the health readers parse gains
`release=` and `engine=` beside the wrapper and the commit (`ops/paper_loop_nightly.sh:124`).
Order effect: none today; on the day the owner sets `PRISM_RELEASE` to a pinned release, what
decides is that release rather than the working tree, and an unresolvable value trades nothing
instead of trading the tree. **Approval status: the mechanism is approved by the section 7
migration row for "The working tree is the live system"; setting the value is the owner's act, and
the specification says so in the same paragraph as this gate: "Restarting or repointing the
scheduler stays an owner act."**

**O15. The four risk-profile ids become files.** `44d8812`, and the section 7 row for "Deployment
thresholds and tighten-only profiles". `research_paper`, `conservative`, `balanced` and
`assertive` become the reference project's files, `research_paper` naming the `paper.toml` that
already carried its name, with the old ids accepted as aliases until supported consumers migrate.
The tighten-only discipline moves with them and is enforced at load, not in the engine. Two
controls have no limit class in section 3.7, the sleeve capital bands and the de-gross arming
flag, so they are named in `INEXPRESSIBLE_CONTROLS` with the reason and stay resolved in code; a
test fails if any field of `DailyBookConfig` or `HedgePolicy` is neither expressible nor reported.
Order effect: a profile that loosens a pin is refused at load rather than silently applied.
Approved (Q5, Q7, Q19).

### 2.2 Not order-affecting, but on the operating path

| # | Change | Stage | Evidence |
|---|---|---|---|
| N1 | Distribution renamed `prism-trading`, version `0.4.0`; import name stays `prism`; a `prism` console script joins the existing entry points | 1 | D1, D13; `pyproject.toml:6-7`; the nightly wrapper already reads `m.version("prism-trading")` at `ops/paper_loop_nightly.sh:118` |
| N2 | Importing `prism` creates no directory, reads no `.env` and touches no credential; commands resolve a workspace, load `.env` and configure logging in `main()` | 1 | `29a6c67`; `src/prism/workspace.py` |
| N3 | `targets_to_orders` delegates to `prism.assembly.size_orders`, so the live protocol and the evaluator size from one implementation | 1 | `src/prism/live/loop.py:206-233`; and the parity gate's 209 of 209 order match is the demonstration that this changed no order |
| N4 | The bundled session table reaches 2006 through 2027; a date outside it raises rather than falling back to weekday arithmetic | 3b | D12; `88b38eb`, validated two-directionally against what SPY traded |
| N5 | `prism check [--expect-session]` and the doctor's new `missed-sessions`, `session-table` and `authorization` rows; `--account` on the doctor | 5 | `183a6bd`, `2aa38c5`. The holiday-blind `weekdays_since` age is deliberately kept as the cross-check, because it needs no table and so cannot be wrong about a year the table does not cover |
| N6 | The provenance stamp gains `release=` and `engine=` | 5 | `ops/paper_loop_nightly.sh:124` |
| N7 | Components declare `scoring_basis` and bars declare `price_basis`; `scoring_basis` enters `config_id` | 1, 3a | D3; the operating momentum book scores on price and is unchanged, its scores pinned bitwise by `tests/test_component_score_parity.py` |
| N8 | CVXPY and Clarabel ship in a `construction` extra and are named in the import hygiene guard | 1, 4 | `5f9ff9d`, verified both ways: green as shipped, and failing with a planted module-level `import cvxpy` |
| N9 | The convex constructor and the factor risk model ship; no arm was promoted, so the operating book is unchanged | 4 | `ef66c9a`: all eighteen intervals straddle zero, smallest two-sided p 0.262, `[promotion_rule] kind = "none"` |

### 2.3 Off the operating path

Listed so they are visibly considered and excluded, not because they need approval here. Each is
research or evidence machinery: the breadth diagnostic loses its viability verdict and keeps the
IR ceiling as a diagnostic (section 7 row 1); the five claim tiers gain display labels and
`gross_edge` drops "N6 gates pass"; registrations move to TOML under `registrations/` with arms
declared and strict mode defined against them (D5); hurdles become excess returns over a declared
cash series compared with a studentized bootstrap (section 7 row); fold tables are frozen and
content-hashed rather than restated as splitter arguments (D7); the canonical fixed-share
accounting produces all new evidence while the constant-weight holding process is retained as
`historical_constant_weight` reproduction mode (section 7 row); imported legacy trial rows carry
an explicit non-identity (D11); and `research_vintage` and `replay_recorded` are not
interchangeable evidence classes (section 7, invariant conflict 1).

### 2.4 What still needs the owner before the switch

Six items. The first two are the ones that change orders.

1. **O8's live wiring.** Decide whether the live decile book moves to `eligibility_exit =
   "reduce_only"` and, if so, when. Leaving it on `exit` is defensible while the heuristic
   constructor is the operating one and no arm was promoted, but the specification's reference
   value from stage 3 is `reduce_only`, and the tree currently has neither the wiring nor a
   recorded reason for the divergence.
2. **`preference_response`.** Spec lines 2846-2851 say the reference value is `record` "while the
   heuristic constructor is the only constructor" and `block` "from stage 4". The tree declares
   `record` (`examples/reference/policies/recovery.toml:10`, `src/prism/template.py:433`,
   `src/prism/portfolio/validate.py:81`), and the specification's own configuration examples at
   lines 3042 and 5356 also show `record`, so the specification is not self-consistent here.
   Either move the reference value to `block` now that the convex constructor exists, or record
   why it stays at `record` while the heuristic constructor is still the operating one. This is
   order-affecting: under `block`, an introduced preference excess refuses the submission.
3. **The authorization record.** Author `authorization.json` for the paper account. Every field is
   required and read strictly: `schema`, `mode`, `account_name`, `account_id`, `endpoint`,
   `execution_host`, `config_id`, `engine_version`, `distribution_id`, `policy_ids` (mirroring
   every key of the `[policy]` block), `scope`, `submission_window_sessions`, `granted_by`,
   `granted_at`, `expires_at` (required and non-null under live, may be null under paper). Until
   one exists the paper loop runs as the pre-authorization instrument, the account lock of O2 does
   not arm, and O3, O4 and the action table are untested by the window.
4. **The declared execution host**, which is the key the lock and the record both bind.
5. **The freshness policy numbers**: `max_broker_state_age_seconds`, `max_clock_skew_seconds`, and
   each source's publication deadline and required coverage of eligible securities. The four
   checks are implemented; the thresholds they compare against are the owner's declaration.
6. **The window's start date and the value of `PRISM_RELEASE`.** Both are owner acts by the
   specification's own words.

Two implementation gaps that `183a6bd` and `e225bcc` left open were closed by `9c9c765` while
this document was being written, and both are worth confirming in whatever release the window
actually runs, because each was a control that existed and was never called. The per-security
freshness verdicts now reach the gate: `decide_and_submit` takes the section 3.11 report and folds
it into the three shapes section 3.10 evaluates, so O6 and O7 act through the decision path rather
than at the gate alone. And the account lock is now armed from the authorization record rather
than from a driver argument no driver passed, so O2 is live wherever a record names this host.
A partial signal outage and a second writer are each proved end to end.

---

## 3. The observation window

### 3.1 Where it starts

The paper loop has been dark since 2026-08-14. Counted in sessions from `prism.io.sessions`,
never in weekdays: the 15 sessions from 2026-08-17 to 2026-09-04 are missed runs, 2026-09-07 is
Labor Day and is a closure rather than a missed run, and 2026-09-08 and 2026-09-09 bring the count
to 17 as of today. The book is a v2 state document holding 99 positions and 937,669.47 in cash,
and its marks are 17 sessions stale.

So the window does not begin with a restart. It begins after one.

### 3.2 Phase 0: the restart, one session, not counted in the window

Do not start the window until all seven hold on a single session:

1. The v2 state document migrates to v3 without loss, compared field by field against the source
   including equity at the prices that book recorded. Run the migration against a copy first; the
   acceptance test at `af35dc8` already does exactly this.
2. `prism check <run-dir> --expect-session` reports the missed sessions honestly and the session
   table answers every date it is asked about.
3. Reconcile against the venue. Positions and cash are adopted and every absorbed difference is
   recorded. Order quantities, sides and unledgered executions are not adoptable and hold the
   engine at reconcile-only; if any appears, stop here.
4. The account-keyed lock is acquired once and released. Note the dependency: it arms from the
   authorization record and only where that record names this host (O2), so without item 5
   below there is no account lock to observe, only the run-directory lock.
5. An authorization record exists for the account and binds this `account_id`, `config_id`,
   `engine_version`, `distribution_id` and execution host (see 2.4 item 3). A window run without
   one observes nothing about the authorization path.
6. The doctor is green, or every red row is understood and named.
7. **The panels the session scored are persisted.** This is the one change without which the
   window ends where the last one did. Verify it by running the gate's own probe after session 1:

   ```
   python -m prism.live.parity_gate --capture <new-capture> --store-dir <store>
   ```

   and confirming the coverage block reports a nonzero count at that session's bar. As of today
   the loop persists nothing (part 1.3), so this needs either a store handed to the bar source or
   a per-session panel snapshot written into the run directory. *Update 2026-09-30:* the
   per-session panel log exists (`prism.live.panel_log`; `runs/<run>/panels/` holds a base
   parquet plus one delta per session, `index.jsonl` indexes them, and each refresh row in
   `targets.jsonl` cites its panel's `panel_sha256` and `panel_file`); what the parity gate still
   needs is to read `panels/index.jsonl` as its coverage source and to open the recorded panel
   when it models fills, instead of asking the bar source's store.

If 1 through 6 hold but 7 does not, the window can still run as an operational soak. It just
cannot close the parity gate, and it should be declared that way up front rather than discovered
at the end.

### 3.3 Length: 43 consecutive trading sessions

Why 43 and not a round number of weeks. The momentum book refreshes every 21 sessions
(`src/prism/scripts/paper_books.py:216` `default_decision_every=21`, and
`ops/paper_loop_nightly.sh:126-132` passes no override), and the first cycle after a restart
always refreshes because the state carries no cadence anchor. So refreshes land on sessions 1, 22
and 43, and 43 sessions is the shortest window that delivers the three decision-bearing sessions
the specification asks parity to be judged on (spec line 5514). Anything shorter leaves the gate
exactly where part 1 leaves it.

Counted in sessions from the bundled table, never in weekdays. A holiday is a closure and is
reported beside the count, not inside it.

Worked example, if the loop restarts on the next session: session 1 is 2026-09-10, session 22 is
2026-10-09, session 43 is 2026-11-09, and the table records no closures inside that span
(Thanksgiving 2026-11-26 falls after it).

### 3.4 What is observed

Eight readings, all of which the engine already produces, so this needs no new instrumentation
beyond phase 0 item 7.

1. **The session ran.** `prism check <run-dir> --expect-session` exits zero. A holiday is a
   closure, not a missed run.
2. **The four freshness checks.** Data freshness per source against its declared publication
   deadline and at its declared coverage; broker-state age against
   `max_broker_state_age_seconds`; clock health against `Broker.server_time` and
   `max_clock_skew_seconds`; scheduler liveness, the last decision bar against the last expected
   session. Record all four separately. A single maximum age is satisfied by one symbol arriving
   on time out of five hundred, which is why there are four.
3. **The authorization verdict per batch.** Which state applied, which action classes were
   permitted, and whether any order request was constructed under a refusal (the answer must
   always be no).
4. **The lock.** One acquisition per session on the declared execution host, released after
   settle. No second acquisition.
5. **Reconciliation.** Every adopted difference recorded; zero unledgered executions; zero
   `unknown_order_state` carried across a session boundary.
6. **The decision and its orders.** Targets, orders, fills, unfilled and equity ledgers written,
   with the write-ahead record preceding the first submit.
7. **The provenance stamp.** `wrapper=`, `commit=`, `release=`, `engine=` on the RUN line, so
   which code traded is on the record and not inferred.
8. **Realized cost against the cost stack**, per the specification's own stage 5 reconciliation
   item (spec line 6108). Reported, not gated: it is an observation, not a promotion test.

### 3.5 Pass

All eight, judged at the end of session 43:

- 43 of 43 expected sessions ran. Zero missed sessions that were not closures.
- Three decision-bearing sessions produced targets, orders and fills, and each has its scored
  panel persisted.
- **The parity gate re-run over a capture of those sessions reports `decision-parity` PASS, with
  3 of 3 decision sessions compared and 0 differing at the declared tolerance.** This is the item
  that closes the gate, and it is the only one that could not be judged in part 1.
- Zero refusals that the action table and the authorization record do not explain.
- Zero unresolved `unknown_order_state` across a session boundary.
- Zero reconciliation differences in the non-adoptable classes: order quantity, side, unledgered
  execution.
- Zero lock contentions.
- The four freshness checks green on every session, or each outage documented with its S-state and
  the consequence it produced.

Realized cost is reported with the pass and does not gate it.

### 3.6 Stop

Any one of these stops the window. Stop means halt the loop, do not advance the live checkout, and
do not repoint the wrappers.

Immediate stop, same session:

- An order reaches the venue that the authorization gate should have refused.
- Any request to a live endpoint under paper mode.
- A second lock acquisition succeeds on the same account.
- An unledgered execution, or an order the venue reports that the engine has no record of.
- A state document that fails to migrate, or a session that starts flat over a non-flat account.

Stop and diagnose before the next session:

- Two consecutive missed sessions that are not closures.
- A decision-bearing session whose orders the gate cannot reproduce from that session's own
  recorded inputs at the declared tolerance. That is the part 1 parity claim failing on fresh
  evidence, which is a stronger signal than any of the checks that passed on the old capture.
- Any behaviour difference that section 2 does not name. An unexplained difference is precisely
  what this gate exists to catch, and the correct response is to add it to section 2 with its
  justification and re-seek approval, not to absorb it.

### 3.7 The switch, in order

1. Section 2.4 items 1 through 5 are decided and recorded, and the loop is configured to match.
   This comes first, not last: the window observes the engine those decisions produce, and an
   item still undecided when session 1 runs is a behaviour the window did not test.
2. Phase 0 passes on one session.
3. The window runs 43 sessions and passes 3.5.
4. The gate is re-run over a capture of those sessions and reports `decision-parity` PASS, so the
   whole report reads PASSED and exits 0.
5. Only then, and as one act: the live checkout is advanced, `PRISM_RELEASE` is set to the pinned
   release, and the section 7 migration's working-tree row takes effect. Restarting or repointing
   the scheduler stays an owner act.

Step 5 is the first irreversible one. Everything before it is reversible by leaving
`PRISM_RELEASE=checkout`, which is what the wrappers default to today.
