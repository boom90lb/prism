# Factory amendment — pipeline-level pre-registration under SPEC §10

> **Status: RATIFIED 2026-07-23** (owner-directed dedicated commit; SPEC §10
> factory clause pasted same commit). `docs/v040_program.md` §5 queue item 4.
> No family may claim "factory" status without its own factory document that
> freezes F-space / procedure / budget B / selection set / cost stack / kill /
> promotion / prior-program count / A4 posture **before** the first counted
> configuration. Ratification of this amendment alone authorizes **no** counted
> industrial run. This document generalizes a shape the X-family already has at
> budget 6; it does not re-open X, trend, momentum, or cert-001, and it moves no
> existing trial value.
>
> Product priority: equity search industrialization first. Event/news
> *modeling* over the observatory expectations lane is a future consumer
> of this doctrine. Capture remains free (W5).

## 0. Problem

SPEC §10 + N5 already require: every searched knob is a counted trial;
budgets are per-family and never refilled; DSR deflates against the
selection-set ledger; one promotion adjudication at a time (A4 for
non-promoting concurrent kills). What the constitution does **not** yet
name is a first-class *pipeline* pre-registration: a frozen feature space,
frozen search procedure, frozen selection-set accounting, and frozen
promotion rule, inside which many configurations may be evaluated
industrially without either (a) inventing a new ad-hoc family document per
knob, or (b) running uncounted search and laundering it later.

Without that name, two failure modes recur:

1. **Budget amnesia** — informal sweeps that "don't count" because they
   were not labeled trials (`docs/handoff.md` §7.2).
2. **Slot begging** — every new idea wants its own full pre-registration
   and its own turn at the serial promotion slot, even when the
   *procedure* is identical and only the configuration changes.

The factory is the procedure. Configurations inside it are the trials.

## 1. What ratifies (SPEC §10 semantics)

On owner dedicated commit, SPEC §10 gains an explicit factory clause
alongside the existing per-family budget and A4 text. Draft wording for
that clause (to be pasted into SPEC at ratification, not before):

> **Factory pre-registration.** A *factory* is a ratified document that
> freezes, before any configuration in its scope is evaluated: (i) the
> feature / signal space; (ii) the search procedure (how candidates are
> generated and ordered); (iii) the selection-set identity and the
> never-refilled budget `B`; (iv) the kill and promotion rules; (v) the
> cost stack and construction pins shared by every configuration. Every
> configuration evaluated under a factory is a counted trial against that
> factory's ledger, including degenerate and NaN outcomes. A factory does
> not create a second adjudication slot: promotion reads still require
> the serial slot; non-promoting reads may use A4 only if the factory's
> banner opts in. Continuous hyperparameter roaming (unbounded search
> without a frozen procedure and budget) remains barred. Existing
> one-shot family pre-registrations (momentum M-series, trend T-series,
> learned-XS X-series, replication C-series) are not retroactively
> factories; they stand as written. A family may *declare* factory shape
> at its own ratification or by seam amendment; declaration does not
> refill budget or re-open closed selection sets.

No other SPEC section is amended by this draft. N1–N8, claim tiers, and
I-1…I-9 stand.

## 2. Factory document contract (what every factory must pin)

A factory pre-registration is not valid (and cannot be ratified) unless
it pins all of the following *before* the first counted configuration:

| Pin | Meaning | Failure if missing |
|---|---|---|
| **F-space** | Enumerated feature / signal atoms or a closed generator with a finite known cardinality at ratification | Open-ended feature invention mid-search |
| **Procedure** | How the next configuration is chosen (grid, fixed list, nested CV schedule) — including stop rules | Continuous roaming |
| **Budget B** | Integer, never refilled; every eval counts | Infinite runway |
| **Selection set** | Namespace + era + universe identity | Ledger amnesia / cross-set laundering |
| **Cost + construct stack** | Spreads, band, participation, cadence pins shared by all cells | Per-cell cost shopping |
| **Kill rule** | STOP condition readable without promotion | Search without exit |
| **Promotion rule** | What may claim `net_edge` / supersession; always serial-slot | Soft bar |
| **Prior programs count** | Number of prior counted programs at ratification (SPEC §10) | Multiplicity opacity |
| **A4 posture** | Opt-in or serial-only, explicit | Implicit concurrency |

Optional but recommended: parity nest (a configuration that must
bit-reproduce a certified sibling), spanning / incrementality read when
the claim is "adds over X", and a firewall clause for A4 outputs.

## 3. Relationship to existing families

| Family | Factory shape today? | Action under this amendment |
|---|---|---|
| residual reversion | closed at 17; cert-001 | **No revival.** Closed selection set stays closed. |
| momentum M-series | one-shot robustness grid, budget 8 | Stands; not retroactively a factory. |
| trend T-series | one-shot fragility grid, budget 6 | Stands; A4 opt-in is a separate seam. |
| learned_xsection X-series | **already factory-shaped** at budget 6 (frozen F0, L0, grid X0–X5) | Stands; may *label* itself factory by optional banner note after this amendment ratifies — label only, no value moves. |
| replication C-series | zero-dof cells, budget 3 | Stands; not a search factory. |
| aim-portfolio | gated construction search | Becomes countable only under its own G4b gate; may adopt factory contract for construction cells when G4b opens. |
| future event/surprise | none | **Must** register as a factory (or one-shot family) before any modeling trial; observatory *capture* does not wait. |

## 4. What the factory is not

- Not a license to soft-bar or to move DSR thresholds.
- Not a shared enterprise budget (A1 already killed that).
- Not permission for research-tree models to enter the production import
  path (N8).
- Not a second promotion slot.
- Not automatic A4 opt-in (banner still required per family/factory).
- Not a way to re-open closed selection sets by renaming them.

## 5. Ratification checklist (owner)

1. ~~Dedicated commit: this banner → RATIFIED; SPEC §10 paste of §1 clause.~~
   **Done 2026-07-23.**
2. No other files in that commit except necessary SPEC cross-links and
   this document's status line. **Done (this commit).**
3. Optional follow-up (separate commit): X-family banner note "this family
   is factory-shaped under `docs/factory_amendment.md`" — label only.
4. No counted run is authorized by ratification alone; each factory still
   needs its own frozen budget and, for promotion, the serial slot.

## 6. Non-goals for this draft

- Implementing industrial runners or new CLI surfaces.
- Event/news model pre-registration (downstream consumer).
- Crypto family pre-registration (queue item 7; equity-first).
- Any change to M6, T5, X5 promotion arithmetic.
