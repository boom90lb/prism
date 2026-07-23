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
| Intraday anything | SPEC §8 | Closed |

---

## 2. Open law junctures (do these before “implement the account”)

These three are queue items 4–6. They are *not* the micro-account itself.
They are the last constitutional/product freezes on the equity path that
the program still marks DRAFT.

### J4 — Factory amendment (SPEC §10)

| | |
|---|---|
| **Decision** | May industrial search exist as a first-class *procedure* with a frozen budget, without inventing a new family doc per knob? |
| **Read** | `docs/factory_amendment.md` end-to-end (short). Focus §1 clause + §2 pin table + §4 “what it is not” |
| **Verify** | (1) Clause freezes F-space, procedure, budget B, selection set, cost/construct stack, kill, promotion, prior-program count. (2) **No second promotion slot.** (3) Existing families are **not** retroactively factories. (4) Closed sets (cert-001) stay closed. (5) Observatory *capture* remains free; event *modeling* still needs its own pre-reg later |
| **Commit shape** | **Only:** banner → RATIFIED on `docs/factory_amendment.md` **and** paste §1 clause into `SPEC.md` §10. Optional later commit: X-family “factory-shaped” *label* only |
| **Do not** | Authorize a counted factory run in the same commit; soft-bar; re-open residual; mix feature code |

### J5 — Aim-portfolio G4a / G4b split

| | |
|---|---|
| **Decision** | Separate *uncounted multi-sleeve engineering* (G4a) from *counted construction search* (G4b)? |
| **Read** | `docs/aim_portfolio_preregistration.md` banner through the split table (~§0 box). Full G-P design only if you will open G4b soon |
| **Verify** | (1) While DRAFT, **hard gate unchanged** — zero counted aim-portfolio trials. (2) G4a opens on ≥2 sleeves at `mechanics_clean`; spends **no** deflation budget; joint crash / fixed-weight sensitivity only. (3) G4b opens only on ≥1 sleeve at its **ratified bar** (or your explicit narrower A4 clause). (4) Optimized multi-sleeve weights are **not** free parameters before G4b freezes them. (5) A4 firewalled kill outputs do not silently become G4b evidence |
| **Commit shape** | Dedicated banner flip DRAFT → RATIFIED for the **split box only**. Do not “accidentally” ratify the whole construction trial table unless you intend G4b law now |
| **Do not** | Treat joint-crash receipts as promotion; treat DRAFT as un-gating; open G4b because 2022 sensitivity looked nice |

### J6 — Risk-profile schema freeze (W6)

| | |
|---|---|
| **Decision** | Freeze the operator surface: named profiles + hedge composition that may only **tighten** pins |
| **Read** | `docs/risk_profile_schema.md` §0–§2 and §4 freeze criteria |
| **Verify** | (1) Profiles: `research_paper` / `conservative` / `balanced` / `assertive` — intent table acceptable. (2) **Tighten-only** vs ratified pins (no g freer than sizing; no gross above certified construction pin). (3) `research_paper` must remain bit-identical to certified B1 paper path (G6). (4) Crypto default **off**. (5) De-gross `armed` is **not** a profile toggle — separate GO arming commit after handoff §8 preconditions. (6) Unknown profile id fails loud |
| **Commit shape** | Banner DRAFT → FROZEN (or RATIFIED). Prefer **no code** in the freeze commit unless a bit-identity test is the freeze receipt |
| **Do not** | Ship profile-aware live deploy before freeze; invent “assertive = more leverage than certified”; arm de-gross via profile |

**Suggested order:** J4 (constitution) → J5 (construction accounting) → J6 (product surface). You may freeze J6 alone if you want operator language before industrial search law; do not open G4b without J5.

---

## 3. Account junctures (after J4–J6, or in parallel only where noted)

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
| Profiles draft | `docs/risk_profile_schema.md` |
| Sizing + arming | `docs/sizing_preregistration.md` |
| A3 micro-account | `docs/amendments_2026-07.md` A3 |
| Regime / 21 sessions | `docs/regime_step.md` |
| Credentials / paper-first | `docs/security.md` |
| Agent conduct | `AGENTS.md` |
| Historical bird's-eye (not this handbook) | `docs/program_review_2026-07.md` |

When this handbook and a banner disagree, **the banner on the binding
document wins** — then fix this file in a follow-up docs commit.
