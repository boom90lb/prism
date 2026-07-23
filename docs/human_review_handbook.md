# Human review handbook — critical junctures only

> **Purpose.** A short owner checklist for decisions that *cannot* be
> delegated to agents. Not an onboarding tour, not a code audit template,
> not a re-statement of SPEC. If a step is missing here, it is either
> already law, or it is agent-safe uncounted work.
>
> **Authority.** SPEC.md → `docs/handoff.md` → this program
> (`docs/v040_program.md`, RATIFIED) → family pre-registrations → AGENTS.md
> (conduct only). On conflict, the higher document wins.
>
> **How to use.** One juncture at a time. Each block: *decision*, *read*,
> *verify*, *commit shape*, *do not*. Skip closed rows. Do not bundle
> ratifications with features.

---

## 0. Standing rules for every owner commit

| Rule | Why |
|---|---|
| **Dedicated commit** per ratification / freeze / arming | History is the lab notebook; mixed commits bury the binding act |
| **Banner flip is the binding act** (not chat, not PR comment) | Agents and future you read banners |
| **No trial values move under a "docs" commit** | Seams and freezes pin procedure; they do not re-search |
| **No bar moves** | Deploy-first re-ranks the *prize*, never the *filter* |
| **No push/PR of law without intent** | Local `main` may hold unpushed constitution; treat push as publication |
| **Real money ≠ paper** | `APCA_API_BASE_URL` unset ⇒ paper (`docs/security.md` §2.5). Live is explicit |

**Status vocabulary you will see:**

| Word | Means for you |
|---|---|
| **RATIFIED** | Law. Do not re-litigate unless a pre-registered re-open trigger fires |
| **DRAFT** | Proposed text. Agents may draft; only you bind |
| **FROZEN** (W6) | Product schema locked; renames need a new dedicated commit |
| **Uncounted** | Engineering / diagnostic. No deflation budget. Still must not invent GO |

---

## 1. Closed — do not re-open in this review pass

| Item | Where | Note |
|---|---|---|
| Constitution N1–N8, claim tiers, kill-criterion shape | `SPEC.md` | Invariants |
| Cert-001 residual daily negative | `docs/certifications/001-…` | Closed selection set |
| Momentum M-series pins | `docs/momentum_design.md` | Promotion only at M6 window |
| Trend family + A4 T0–T4 | `docs/trend_design.md`, `docs/amendments_2026-07.md` A4 | T5 still serial |
| Learned-XS family + A4 X0–X4 | `docs/learned_xsection_design.md` | X5 still serial |
| Sizing pins (θ, g, hysteresis, floors) | `docs/sizing_preregistration.md` | **Arming ≠ ratification** (§5) |
| Program ranking + equity-first | `docs/v040_program.md` | Queue 1 done |
| Factory pre-registration (SPEC §10) | `docs/factory_amendment.md` | **J4 RATIFIED 2026-07-23** |
| Aim-portfolio G4a/G4b split | `docs/aim_portfolio_preregistration.md` (split box) | **J5 RATIFIED 2026-07-23**; G4b table still gated |
| Risk-profile schema (W6) | `docs/risk_profile_schema.md` | **J6 FROZEN 2026-07-23**; field renames need new commit |
| Intraday anything | SPEC §8 | Closed |

---

## 2. Open law junctures

Queue items 4–6 (J4–J6) are **closed**. No further constitutional freezes
on the equity operator path for v0.4.0 product surface. Account junctures
(§3) are owner acts. Do not open G4b without a sleeve at bar.

---

## 3. Account junctures (profile-named GO now legal *as product label*; capital still owner)

These are **owner acts**. Agents may draft checklists and code; they do not
fund, point live URLs, or arm de-gross.

### A — A3 micro-account (cost calibration, not GO)

| | |
|---|---|
| **Decision** | Fund a real-money micro-account solely as the I-9 cost-measurement instrument (SPEC §10 carve-out / amendments A3) |
| **Read** | `docs/amendments_2026-07.md` A3; SPEC §10 micro-account paragraph; `docs/account_size_floor.md` (diagnostic, not law) |
| **Verify** | Cap and purpose are cost calibration, **not** a claimed-edge deploy. Size change is an owner act recorded somewhere durable (commit note or ops log). Equity venue first |
| **Do not** | Conflate A3 fills with `net_edge` promotion; run crypto as the calibration path by default |

### B — Venue path (fractional / short / capital mode)

| | |
|---|---|
| **Decision** | Which admitted capital mode at the intended AUM (whole-share floor vs fractional; short/locate posture) |
| **Read** | `docs/sizing_preregistration.md` §2 venue checks + §7 parameter table; account_size_floor citations therein |
| **Verify** | Both venue checks pass for the chosen mode, or deployment is blocked (sizing re-open trigger §6.3). Whole-share floor pin is $100k unless fractional path is admitted |
| **Do not** | Pick a mode that fails short/fractionability and “hope paper is enough” |

### C — GO preconditions (handoff §8) — both required

| # | Precondition | Evidence to demand |
|---|---|---|
| **(a)** | Sizing pre-registration **ratified** with crash-conditional de-gross | Banner on `docs/sizing_preregistration.md` (done 2026-07-20). Pins stand; **hook still unarmed** |
| **(b)** | SPEC §7.7 regime step wired in live cycle **and** ≥ **21 consecutive** clean paper sessions (zero named block failures) | `docs/regime_step.md`; run-dir `regime.jsonl` / ledger tools in `prism.live.loop`; no N7 in that streak |

A “GO read” with (a) or (b) unmet **deploys nothing** — it starts wiring, not the account (`docs/handoff.md` §8).

### D — De-gross arming (separate from sizing ratification)

| | |
|---|---|
| **Decision** | Arm the §7.7 gross-scale hook with ratified θ / g |
| **Read** | `docs/sizing_preregistration.md` §5 (arming is *not* the banner flip) |
| **Verify** | (a) and (b) both hold. Paper stream used for (b) was **not** already de-grossing (would fork the certified convention). Safety halt beats de-gross |
| **Commit shape** | Dedicated GO-branch arming commit. Nothing else |
| **Do not** | Arm in the same commit as feature work or profile freeze |

### E — First real-money GO order (only if a sleeve cleared its bar)

| | |
|---|---|
| **Decision** | Place capital under a **named** risk profile at minimum viable size |
| **Read** | Family promotion section (e.g. momentum §3, trend §4); claim packet; capacity / participation pins; frozen W6 profile |
| **Verify** | Sleeve at bar under its own ledger. Profile id recorded. Live monitor / kill-switch posture known. `APCA_API_BASE_URL` intentionally live. Scale only along capacity curve |
| **Do not** | GO on uncounted joint-crash warmth; GO on A3 micro fills; GO single-sleeve without the correct product label (`single-premium, de-gross-only` vs preferred multi-sleeve) |

---

## 4. Permanent code junctures (re-review whenever touched)

These paths are always load-bearing. A PR that changes them needs human
eyes even when “just a refactor.”

| Surface | Path (representative) | Why it is critical |
|---|---|---|
| Trial ledger | `results/stat_arb_residual_trials.jsonl` (+ family ledgers) | Append-only; delete/rewrite = falsification |
| Next-open fills / costs | `src/prism/execution/target_weights.py`, `costs.py` | Economic truth of every backtest |
| Live cycle | `src/prism/live/daily.py`, order/reconcile path | Real money when pointed live |
| Broker env | `src/prism/live/alpaca.py` (`from_env`) | Paper default vs live URL |
| Import boundary | `tests/test_import_hygiene.py`, `src/prism/` vs `research/` | N8 + research quarantine |
| Claim packets / certs | `results/**/claim_packet.json`, `docs/certifications/` | Public honesty surface |
| Regime seam | `docs/regime_step.md`, live regime ledger | GO precondition (b) |
| Construction combine | aim-portfolio docs + any sum-then-cap remnant | Anti-netting failure mode (cert-001 §8) |
| Risk profiles (W6) | `src/prism/live/risk_profile.py`, run-dir `profile.json` | Product surface FROZEN; may only tighten pins |

**Agent-safe (usually no owner commit):** uncounted diagnostics, capture-only
observatory, trend/paper mechanics default-off, joint-crash arithmetic,
docs drafts under DRAFT banners.

---

## 5. One-page session checklist (print / paste)

```text
[ ] Which juncture? J4 / J5 / J6 / A / B / C / D / E  (one)
[ ] Read list finished; conflicts with SPEC/handoff noted
[ ] Verify bullets all pass (or explicit defer with reason)
[ ] Commit message: docs(constitution|product|go): … RATIFIED|FROZEN|ARMED
[ ] Commit contains only the binding files for that act
[ ] No trial ledger edits; no bar language softened
[ ] If real money: paper URL not assumed; profile named; (a)+(b) true
[ ] Next juncture written down; agents unblocked on uncounted work only
```

---

## 6. Pointer index (do not duplicate law here)

| Topic | Canonical file |
|---|---|
| Program + queue | `docs/v040_program.md` §5–§7 |
| GO preconditions | `docs/handoff.md` §8 |
| Factory draft | `docs/factory_amendment.md` |
| G4 split draft | `docs/aim_portfolio_preregistration.md` (banner box) |
| Profiles (frozen) | `docs/risk_profile_schema.md` |
| Sizing + arming | `docs/sizing_preregistration.md` |
| A3 micro-account | `docs/amendments_2026-07.md` A3 |
| Regime / 21 sessions | `docs/regime_step.md` |
| Credentials / paper-first | `docs/security.md` |
| Agent conduct | `AGENTS.md` |
| Historical bird's-eye (not this handbook) | `docs/program_review_2026-07.md` |

When this handbook and a banner disagree, **the banner on the binding
document wins** — then fix this file in a follow-up docs commit.
