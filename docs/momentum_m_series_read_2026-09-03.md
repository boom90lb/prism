# Momentum M-series fragility read — 2026-09-03

**Kind:** read record for the registered fragility clause of
`docs/momentum_design.md` §3. **Counted trials:** M1–M5, run 2026-09-03
01:59–02:03 PDT (ledger timestamps UTC) at commit `2b4bca9`: tracked tree
clean, 12 untracked paths (root dotfiles and this read script), none imported
by the driver. Each run ~1 minute, sequential, deterministic ledger order.
**Ledger:** `results/momentum_v1_trials.jsonl`, 6 rows: the B1 discovery
record imported as M0 (a manual byte copy of row 16 of the residual ledger),
then M1–M5. **Budget:** 8 registered; 6 spent; M6 (extension read, ≥ 2027-06
data) reserved; M7 reserve unspent. **Read script:**
`research/scripts/momentum_fragility_read.py` →
`results/momentum_m_series_read_2026-09-03.json`.

## Verdict: SURVIVES the fragility read

- Clause (a): median net annualized Sharpe of M1–M5 = **0.477** (B1: 0.465); the
  minimum is 0.324. Not negative.
- Clause (b): no single knob move flips the sign of the net result; all five
  cells are net-positive, so no magnitude test is reached. The same test on net
  total return finds no flip either, and the largest swing from B1 (M2, −0.14
  Sharpe) is below B1's point estimate under the "delta" reading as well.

Survival is not promotion. Promotion reads only at M6 + the paper stream (§3).

**§4 judgment ("a wildly positive robustness set should raise suspicion"):**
flat, not wildly positive. Three cells are above B1 by at most 0.09 Sharpe
(M1 +0.01, M5 +0.06, M4 +0.09) and two below (M3 −0.06, M2 −0.14), in the
same sample in which B1 was the best of 17 mostly non-momentum trials. Per
probe: the skip month is not load-bearing (M1 ≈ B1; M5, a 2-month skip, is
slightly higher); the 6−1 horizon is the weak axis (M2); quintiles dilute
(M3); quarterly cadence is at least as good and cheaper (M4).

## Results

| trial | delta vs B1 | net ann. Sharpe | net total return | gross total return | total cost | turnover/day | max DD | arith. mean return /yr | clears 3.71 % T-bill | tier |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 = B1 (import, not re-run) | B1 (M0 import) | 0.465 | 27.6 % | 32.4 % | 3.70 % | 0.050 | 14.0 % | 5.36 % | yes | net_edge |
| M1 | mom_skip_bars=0 | 0.477 | 29.1 % | 34.0 % | 3.69 % | 0.050 | 14.7 % | 5.63 % | yes | net_edge |
| M2 | mom_lookback_bars=126 | 0.324 | 16.3 % | 21.2 % | 4.18 % | 0.060 | 12.9 % | 3.49 % | no | net_edge |
| M3 | mom_decile=0.2 | 0.409 | 18.0 % | 22.2 % | 3.53 % | 0.047 | 11.1 % | 3.56 % | no | net_edge |
| M4 | decision_every=63 | 0.552 | 32.3 % | 36.1 % | 2.81 % | 0.032 | 13.4 % | 5.99 % | yes | net_edge |
| M5 | mom_skip_bars=42 | 0.530 | 30.7 % | 35.7 % | 3.75 % | 0.051 | 13.5 % | 5.75 % | yes | net_edge |

The T-bill column is the plain form of the registered periodic-hurdle test,
which is vol-invariant (hurdle = annual/252 ÷ daily vol, so pass/fail is a
mean-return comparison): M2 and M3 earn less than the 3.71 % bill basis, the
other cells and B1 earn more. §3's fragility clause does not read the hurdle
(it is a promotion conjunct at M6), so this has no registered consequence; it
is the informative sensitivity in the set. `docs/momentum_design.md` §0's
"the only cash-hurdle-clearing result in program history" is now historical:
M1, M4 and M5 clear it in the same sample (the certification text is immutable
and is not edited).

## N6 breadth / falsification diagnostic (uncounted)

| trial | N_eff | rank IC (h = cadence) | IC se | n windows | ceiling IC·√N_eff | realized net | realized gross | flag net | flag gross | capture net / gross |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 = B1 | 52.7 | 0.0302 | 0.0241 | 62 | 0.219 | 0.134 | 0.152 | no | no | 0.61 / 0.69 |
| M1 | 51.5 | 0.0303 | 0.0242 | 62 | 0.217 | 0.138 | 0.155 | no | no | 0.63 / 0.71 |
| M2 | 69.4 | 0.0101 | 0.0233 | 62 | 0.085 | 0.093 | 0.115 | **yes** | **yes** | 1.11 / 1.36 |
| M3 | 55.3 | 0.0302 | 0.0241 | 62 | 0.224 | 0.118 | 0.141 | no | no | 0.53 / 0.63 |
| M4 | 49.4 | 0.0411 | 0.0301 | 20 | 0.289 | 0.276 | 0.301 | no | **yes** | 0.96 / 1.04 |
| M5 | 54.7 | 0.0332 | 0.0237 | 62 | 0.245 | 0.153 | 0.172 | no | no | 0.62 / 0.70 |

`research/scripts/breadth_diagnostic.py` per run dir (`breadth_diagnostic.json`,
committed), run because §4 points at the §10 falsification gate. N6 reads
"realized above ceiling ⇒ leak or bug". Two cells sit at or above their
point-IC ceiling: **M2 on net and gross** (IC 0.010 ± 0.023, n = 62) and **M4
on gross** (IC 0.041 ± 0.030, n = 20 at the 63-bar horizon; net capture 0.96).
What was checked for a leak or bug: (i) the uncounted reproduction control
below shows this commit reproduces the certified B1 packet bit-for-bit, and B1
sits at 61 % of its ceiling; (ii) each M cell differs from B1 by one knob on
the same code path, and a look-ahead in that path would lift every cell, not
the two with the least precise IC estimates; (iii) the momentum node's
causality is property-tested (`tests/test_momentum_node.py`); (iv) at IC + 1 se
both ceilings clear their realized values by a wide margin. Judgment: no leak
or bug found; the flags measure the imprecision of a ceiling estimated from a
near-zero IC, and the same imprecision means the "no" flags on the other cells
are not decisive. Recorded as an open diagnostic, not a pass. Viability at the
one-sided lower-95 IC fails for every cell, as for B1.

## Deflation (not an adjudication input)

| trial | DSR as written (sequence-dependent, rows present at run time) | DSR uniform vs final 6-row ledger, N = 8 |
|---|---|---|
| M0 = B1 | 0.191 | 0.781 |
| M1 | 0.853 | 0.788 |
| M2 | 0.674 | 0.676 |
| M3 | 0.756 | 0.742 |
| M4 | 0.833 | 0.834 |
| M5 | 0.822 | 0.822 |

The driver deflates each row against the ledger rows present when it ran
(M1's benchmark was two nearly identical Sharpes), so the as-written values are
sequence-dependent; the uniform column recomputes every row against the
finished ledger at N = 8. Either way these are fragility rows in an 8-trial
namespace; they promote nothing, and B1's promotion DSR remains the 0.191 of
its 17-trial discovery set until M6.

## Pinned readings (interpretation, no trial value moved)

- "Net result" in clause (b) is the statistic clause (a) uses: net annualized
  Sharpe under bucket spreads.
- M4's registered delta `decision_every=63` sets both cadence fields
  (`walk.mom_decision_every` and `signal.decision_every`; B1 had both at 21),
  so the closed-form band's variance input follows the cadence. One knob.
- `--design_trials 8` (the registered budget) is the deflation denominator.
- Ledger rows carry the driver's hardcoded strategy tag `residual_wfo_v1` and
  packets `residual_stat_arb_wfo`; §2's `momentum_v1` namespace is realized as
  the ledger file, which is what deflation reads. The ledger `config_hash` and
  the packet `config_hash` are different digests of one configuration, as for
  B1.

## Reproduction control (uncounted)

`results/momentum_m0_repro_2026-09-03`: B1's exact flags under commit
`2b4bca9` reproduce the certified packet bit-for-bit (Sharpe
0.46544245642237375, total return 0.2758, cost 0.0370, turnover 0.0500, 574
names, 21 folds; config diff empty). Scratch ledger; not in the momentum ledger
and not a budget row (same configuration, no search).

## Verification performed

Tested/Run, this session, inline: each trial's `config.json` differs from B1's
in exactly the registered knob plus `design_trials` (now asserted by the read
script); Sharpe and total return recomputed from `returns.csv` equal
`summary.json` to four decimals; `validate_claim_packet_dir` passes on all six
dirs; ledger row 1 is byte-identical to residual-ledger row 16, the ledger's
output_dirs equal the registered set in order (asserted by the script), and
timestamps are monotone; the residual ledger (17 rows) and `results/demotion_b1`
are unchanged. Then a blind workflow: two independent re-derivations from the
registration text and raw packets (both SURVIVES, identical arithmetic, one
maximally adversarial toward SURVIVES), a run-fidelity audit (faithful), a
completeness critic, and a reconciliation. Corrections adopted from it: gross
total return is now the packet's compounded figure (the first script version
added net and cost); M4's gross-side N6 flag is reported; the hurdle failures
are stated as the mean-return fact they are; the DSR sequence dependence is
recorded; SPEC §1 was updated.

## Guardrails

No configuration moves. Adopting a different cell is a new discovery event
(§2); changing the live book's configuration is an owner act (SPEC §6); M7 is
reserved for one *seam*, and a cadence follow-up prompted by M4 would be a
search, hence a new registration. M4's improvement decomposes into gross
(36.1 % vs 32.4 %) and cost (2.81 % vs 3.70 %; turnover 0.032 vs 0.050/day),
with its gross above its own point-IC ceiling.

## What next

M6 stays on its clock. The promotion conjunct's live-monitor read and the
21-session clock are frozen: the paper stream stopped at 2026-08-14 because
the scheduler stopped firing (SPEC §1). T0–T4 follow once the counted trend
driver exists (SPEC §5).

## Command lines

Common: `--universe data/universe/sp500_pit_resolved_2026-06-16.txt --membership
data/universe/sp500_membership_2026-06-16.parquet --coverage
data/universe/sp500_coverage_2026-06-16.json --start_date 2020-01-01 --end_date
2026-06-16 --formation_bars 312 --test_bars 63 --min_test_bars 20 --band_mode
closed_form --spread_mode bucket --max_participation 0.05 --max_gross 1.0
--max_symbol_abs_weight 0.35 --sleeve_mode momentum_only --hurdle_annual_pct
3.71 --hurdle_basis tbill_nominal --design_trials 8 --trial_ledger
results/momentum_v1_trials.jsonl`, then per trial `--decision_every 21
--mom_lookback 252 --mom_skip 21 --mom_decision_every 21 --mom_decile 0.10`
with the registered delta applied (M4: both cadence flags 63).
