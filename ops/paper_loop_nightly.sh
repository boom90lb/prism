#!/usr/bin/env bash
# Nightly Alpaca paper-loop runner — the I-9 cost-measurement instrument
# (SPEC §13 R2; prism.scripts.paper_loop). Scheduled weekdays 18:30 PT
# (21:30 ET — after the close, well before the ~09:28 ET OPG cutoff next
# morning); the current operator drives it from Windows Task Scheduler via
# wsl.exe.
#
# Versioned in-tree as of the P0 operational recovery. It used to live only at
# ~/bin/prism_paper_loop_nightly.sh, which is why its two defects were invisible
# to review: it inferred nothing from a nonzero exit, and nothing else did
# either. Six consecutive nightly failures (2026-07-23..29) reached nobody.
#
# Book: the ratified B1 momentum candidate (12-1 cross-sectional momentum,
# decile long/short, monthly cadence — docs/momentum_design.md §3), cut over
# 2026-07-09. The live-monitor read is a B1 concordance read only under this
# book. Both books share one paper account and must not trade it concurrently.
#
# Safe to re-run on the same bar: the write-ahead protocol resumes the
# persisted decision and duplicate client_order_ids settle as DuplicateOrder
# (prism/live/loop.py), so a rerun never double-trades.
#
# ARMING: paste the Alpaca paper credentials into $REPO/.env (same file as
# TWELVEDATA_API_KEY):
#   APCA_API_KEY_ID=...
#   APCA_API_SECRET_KEY=...
# Until then this script logs SKIP and exits 3.
set -u

REPO="${PRISM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Single switch shared with the morning sweep: runs/ACTIVE_RUN_DIR names the
# live run-dir (one line, relative to the repo).
RUN_DIR="$(cat "$REPO/runs/ACTIVE_RUN_DIR" 2>/dev/null || echo runs/paper_loop_momentum)"
LOG_DIR="$REPO/$RUN_DIR"
LOG="$LOG_DIR/nightly.log"
mkdir -p "$LOG_DIR"
ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }
# shellcheck source=ops/_alert.sh
. "$REPO/ops/_alert.sh"

# prism.config's load_dotenv() will read $REPO/.env on import; source it
# here as well so the credential gate below can decide without Python.
set -a
# shellcheck disable=SC1091
[ -f "$REPO/.env" ] && . "$REPO/.env"
set +a

if [ -z "${APCA_API_KEY_ID:-}" ] || [ -z "${APCA_API_SECRET_KEY:-}" ]; then
    echo "$(ts) SKIP: APCA_API_KEY_ID/APCA_API_SECRET_KEY not present in $REPO/.env — paste the Alpaca paper keys there to arm the loop" >>"$LOG"
    exit 3
fi

# Universe: current S&P 500 members with fetchable bars
# (data/universe/sp500_current.txt) — decile 0.10 gives a ~50/50 L/S book.
# B1 knobs (lookback 252, skip 21, decile 0.10, cadence 21, max-missing 0.10)
# are paper_loop's momentum-mode defaults; do not restate them here. Names the
# book holds that have since left the index are fetched anyway, from broker
# truth, for valuation and exit (prism.live.daily.resolve_fetch_universe).
UNIVERSE="data/universe/sp500_current.txt"

# Regime telemetry (SPEC §7.7 step — GO precondition (b)'s clean-session
# clock, docs/regime_step.md): self-arms when FRED_API_KEY is present in
# $REPO/.env; the count starts the first night this logs regime=on.
# Telemetry only — the gross-scale action hook has no CLI path until
# docs/sizing_preregistration.md ratifies.
REGIME_FLAG=""
REGIME_STATE=off
if [ -n "${FRED_API_KEY:-}" ]; then
    REGIME_FLAG="--regime"
    REGIME_STATE=on
fi

cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python3"
echo "$(ts) RUN: book=momentum universe=$UNIVERSE regime=$REGIME_STATE" >>"$LOG"
# shellcheck disable=SC2086
"$PY" -m prism.scripts.paper_loop \
    --book momentum \
    --universe-file "$UNIVERSE" \
    --run-dir "$RUN_DIR" \
    --spinoff-mask \
    $REGIME_FLAG \
    >>"$LOG" 2>&1
rc=$?
echo "$(ts) EXIT $rc" >>"$LOG"

# The health verdict, not the exit code, is what the operator is told about.
# A cycle can exit 0 and still leave the book unhealthy for tomorrow (an
# unattributed venue book, a held position with no bars, a stale NAV ledger),
# and those are exactly the conditions that produced this outage — so the
# doctor runs after every session, red or green, and its exit code is part of
# the alert condition. --network reconciles the venue's book against this run
# directory: two free Alpaca reads and one Twelve Data credit.
echo "$(ts) DOCTOR:" >>"$LOG"
"$PY" -m prism.scripts.doctor --run-dir "$RUN_DIR" --universe-file "$UNIVERSE" --network >>"$LOG" 2>&1
doctor_rc=$?
echo "$(ts) DOCTOR EXIT $doctor_rc" >>"$LOG"

if [ "$rc" -ne 0 ] || [ "$doctor_rc" -ne 0 ]; then
    alert "$LOG" "$LOG_DIR" \
        "prism nightly unhealthy (loop=$rc doctor=$doctor_rc)" \
        "run-dir $RUN_DIR; read the FAIL lines and the traceback at the tail of $LOG. Failed sessions cannot be backfilled as live evidence (docs/operations.md)."
else
    clear_alert "$LOG" "$LOG_DIR"
fi

# Off-box artifact sync runs unconditionally — a red loop is exactly when the
# ledger copy matters most (docs/security.md par.3), and it carries the ALERT
# marker off the box with the ledgers. Operator-specific, hence an override
# rather than a path baked into a versioned file: PRISM_ARTIFACT_SYNC=/dev/null
# (or any non-executable path) disables it.
SYNC="${PRISM_ARTIFACT_SYNC:-$HOME/bin/prism_artifacts_sync.sh}"
[ -x "$SYNC" ] && "$SYNC"

# Nonzero on either verdict: the scheduler's own last-result surface is the
# third independent notification path (ops/_alert.sh).
[ "$rc" -ne 0 ] && exit "$rc"
exit "$doctor_rc"
