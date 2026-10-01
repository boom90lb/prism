# Live monitor: return and tracking statistics (`live_monitor_v1`)

**Status: registered 2026-09-30.** The registration is the commit that adds
this file (SPEC §4). It pins, before any qualifying observation exists, the two
statistics the nightly computes over the paper book's ledgers, their exact
definitions, thresholds and sample floors, and what "the live monitor is not in
contradiction" means for registrations made after this date. Nothing here gates
an order or moves a recorded verdict.

## 1. What the previous monitor could and could not resolve

The monitor in place until this date formed a NAV return over every adjacent
pair of equity-ledger rows, so the six-week outage from 2026-08-14 to
2026-09-25 entered as one "daily" return, and fed a normal-mixture confidence
sequence whose width is set by the assumed support (|r| ≤ 0.1), not by the
realised variance. Logged on 2026-09-29: n = 19, interval about ±0.21 per day
around the mean. Daily NAV volatility is about 0.7%, so a book losing at a
Sharpe of −1 (−0.044% a day) would have read "inconclusive" for a decade.

No return-level monitor can do much better. The half-width of even a
fixed-sample 95% interval on the mean daily return after T sessions is about
2σ/√T; at σ = 0.7% and T = 252 that is 0.09% a day, about two annualized
Sharpe units, so a 0.4-Sharpe book cannot be resolved in a year on its returns.
What can be resolved quickly is the paper book's return minus the return of the
book it was supposed to hold. That difference has a dispersion of a few basis
points a day, and it answers the question the return monitor was being asked to
answer: is the paper stream evidence about the strategy at all?

## 2. The two statistics

Both are computed by `prism.live.monitor` from the run directory's ledgers and
logged by the nightly after the NAV mark. Both use the empirical-Bernstein
confidence sequence of Waudby-Smith and Ramdas (2023, *Estimating means of
bounded random variables by betting*, JRSS-B, Theorem 2, predictable plug-in
form), implemented in `prism.validation.anytime.empirical_bernstein_confidence_sequence`:
observations winsorised to [−b, b] and mapped to [0, 1]; bets
λ_i = min(√(2 log(2/α) / (σ̂²_{i−1} · i · log(i+1))), 3/4) from the running
mean and variance; interval Σλ_iX_i/Σλ_i ± (log(2/α) + Σ v_i ψ_E(λ_i))/Σλ_i with
v_i = 4(X_i − μ̂_{i−1})² and ψ_E(λ) = (−log(1−λ) − λ)/4. Coverage is 1 − α
simultaneously over all sample sizes and nonasymptotic for bounded
observations, so the reads may be inspected every session and acted on at any
data-dependent time. Its width adapts to the realised variance but keeps a
floor of about 2b·log(2/α)/(0.75·n) in return units: the price of a guarantee
that assumes only a bounded support. α = 0.05 throughout.

**Return monitor.** The NAV return r_t = E_t/E_{t−1} − 1 is formed only over
ledger rows whose decision bars are consecutive exchange sessions
(`prism.io.sessions.next_session(prev) == cur`, and `prev` itself a session).
Every other adjacent pair is a gap, counted (`n_gaps`) and listed
(`skipped_pairs`), never averaged in. Support b = 0.1. The verdict against a
per-session hurdle (default 0) is `confirmed` when the sequence lies entirely
above it, `refuted` when entirely below, else `inconclusive`. The normal-mixture
sequence stays available as `method="normal_mixture"` for comparison; the
default is the empirical-Bernstein one. At the live book's scale (σ ≈ 0.7%,
n = 252) its half-width is about 0.4% a day against the mixture's 2.3%: better,
and still, by the arithmetic above, not a test of a 0.4-Sharpe edge. Its job is
to catch a book that is losing fast and to be the registered "not refuted"
conjunct.

**Shadow return and tracking monitor.** For the session ending at bar t with
previous session t−1, the *baseline* is the latest refresh book in
`targets.jsonl` decided strictly before t−1 (`refresh_bar < t−1`), the book the
account should have carried through the whole session. The *shadow return* is
Σ_i w_i (C_i,t / C_i,t−1 − 1) over the baseline's names priced on both bars in
the panel the cycle scored (`prism.live.monitor.shadow_session_return`). Names
unpriced on either bar are left out and counted; when more than 10% of the
baseline's gross is missing the shadow is refused and no row is written. A
session with no previous consecutive session or no baseline records no row; the
reason is logged. A session whose previous bar is a refresh bar is a
*transition*: the refresh's next-open fills switched the held book mid-session,
so paper minus shadow there is the realised *execution wedge* of that refresh
(reported per refresh, keyed by that refresh's bar, which is the transition
session's previous bar; the row's own `refresh_bar` names the baseline book),
not tracking error.

Over the non-transition sessions joined by decision bar (the shadow row's
`prev_bar` must equal the equity pair's previous bar, else the row is counted
`n_unmatched`), the tracking read reports the confidence sequence on the mean
daily difference d_t = paper_t − shadow_t with support b = 0.002 (clipped days
are counted), its n, the Pearson correlation and the least-squares slope of
paper on shadow, and the verdict:

- `contradicted` when the sequence lies entirely below −0.0002 per day (about
  −5% a year, far beyond unmodelled cash, dividend or rounding effects), or
  when n ≥ 42 and the correlation is below 0.90;
- `consistent` when the sequence lies within [−0.0002, +0.0002], n ≥ 42 and
  the correlation is at least 0.90;
- `inconclusive` otherwise. A sequence entirely above +0.0002 is reported
  (`above_tolerance`) and is a defect to investigate, not a verdict.

Power, stated before any data: with b = 0.002 and the sequence's λ cap of 0.75
the half-width is at least 2b·log(2/α)/(0.75·n) ≈ 0.0197/n, so `consistent` is
unreachable before roughly 100 sessions even with zero dispersion, and a
constant shortfall d first fires `contradicted` at about n = 0.0197/(|d| −
0.0002): about 66 sessions for 5 bp a day (−12% a year), about 100 for 4 bp,
about 200 for 3 bp (`tests/test_live_monitor.py` pins the 66). Realised
dispersion of the daily difference moves the typical detection later (median
about 80 sessions for a 5 bp shortfall at 5 bp of daily dispersion); the n ≥ 42
floor below belongs to the correlation rule, not to this one.

The correlation floor is a fixed-threshold screen on the point Pearson
correlation of all steady sessions to date, re-read every session. Unlike the
confidence sequence it carries no time-uniform error control: the anytime-valid
guarantee stated above applies to the sequence conjunct only. Its false-trip
profile, stated before data: for a book whose daily paper-minus-shadow noise is
at the few basis points assumed here (ρ about 0.99) it does not fire; at
ρ = 0.95 (about 23 bp a day of noise orthogonal to the shadow, against a 70 bp
book) the probability of ever reading below 0.90 over sessions 42–252 is about
2%; at ρ = 0.92 (about 30 bp a day) the single-look rate is about 22% at
n = 42, 8% at n = 126 and 3% at n = 252. The verdict is recomputed from the
ledgers each session and carries no state, and a registration that names it
reads the verdict as of its read date, so the operative false-trip rate is the
single-look rate at that date's n, not a stopping-rule rate. The correlation
floor is what catches a book that holds the wrong names; the slope and the
concordance ledger's gross ratio are what catch a book that holds the right
names at the wrong size, which a correlation cannot see.

## 3. Data

`equity.jsonl` (one NAV row per completed session), `targets.jsonl` (one row per
refresh: weights, reference prices, panel digest), `shadow.jsonl` (one row per
shadowed session: `decision_bar`, `prev_bar`, `shadow_return`, `paper_return`,
`transition`, `refresh_bar`, `n_names_used`, `n_names_missing`,
`gross_missing`; written by the cycle, idempotent per bar), and `panels/` (the
panel each session scored, from 2026-10-01, which lets any shadow row be
recomputed). The first shadow row can land on 2026-10-01: its previous session
2026-09-30 is in the ledger and the 2026-09-25 refresh predates it. The live
book's next refresh is on or about 2026-10-26, so the first transition session
and the first execution-wedge figure come the session after it. The health read
`panel-record` (`prism-doctor`) fails when the panel record lags the ledger. The
health read `shadow-record` (`prism-doctor`) fails when `shadow.jsonl` lags the
ledger on a session that should have produced a row, and warns with the reason
on a session that could not (no earlier ledger session, a gap before it, or no
refresh decided before its previous bar): a shadow path that fails every
session must surface as a health failure, not as a WARNING line in a log that
§1 shows went unread for six weeks.

## 4. What "not in contradiction" means

For the registrations made before this date that name the live monitor
(`docs/forward_design.md`, `docs/momentum_design.md`, and the clocks
`forward_read.py` reads for `docs/earnings_design.md` and
`docs/buffer_design.md`), the registered wording stands: the conjunct is the
return verdict alone, and `refuted` is the contradiction. Those files are
immutable registrations and this one does not amend them. The statistic that
wording resolves to does change here, and this paragraph says so: from
2026-09-30, and before any of their scheduled reads (the first is 2027-03-29),
`forward_read.py` reads that conjunct with the empirical-Bernstein return
sequence over consecutive sessions at the parameters pinned in §2 (b = 0.1,
α = 0.05, hurdle 0 per session), pinned by name in
`forward_read.MONITOR_METHOD` and recorded with its parameters in the read's
packet. The normal-mixture sequence over every adjacent ledger pair, in force
when those registrations were made, is retired because the 2026-08-14 to
2026-09-25 outage entered it as one daily return. The empirical-Bernstein
sequence is narrower (on 2026-09-30, about ±0.062 against ±0.25 per day on the
same 16 returns), so `refuted` is reachable sooner, which only makes promotion
harder. The tracking verdict does not enter those clocks' conjunct.

For any registration made after 2026-09-30 that names the live monitor, "the
live monitor is not in contradiction" means all of: the return verdict is not
`refuted`; the tracking read is present with n ≥ 42 joined non-transition
sessions (`TRACKING_MIN_SESSIONS`, the floor at which the correlation criterion
can fire), and its verdict is not `contradicted`. A tracking read that is
absent, `unavailable`, or below the floor fails the conjunct: the read records
it and the set waits; it does not count the paper stream as evidence. Each is
read from the ledgers as of the read date with the parameters pinned here.

## 5. What the tracking statistic cannot see

The shadow is the decided book marked on the same closes, so a paper stream
that tracks it faithfully says only that the account held what the signal
decided. A wrong signal, a leak in the panel, or a construction defect is
reproduced exactly by the shadow and the tracking read calls it `consistent`.
Those are the return monitor's and the research harness's questions, at their
own, much slower, rate. The shadow also inherits the panel's prices: a vendor
revision shows up as a difference, and a holiday row or a gap removes the
session from both reads rather than entering either.
