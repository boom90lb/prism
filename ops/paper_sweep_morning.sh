#!/usr/bin/env bash
# Morning completion sweep for the paper loop (prism.scripts.paper_sweep;
# docs/momentum_design.md S5 amendment 2026-07-11). Scheduled weekdays 06:50 PT
# (09:50 ET — after the opening auction is terminal). Re-submits the pending
# decision's terminal unexecuted residuals as DAY market orders under :S1 client
# ids; the evening settle ledgers auction and sweep fills alike against the same
# decision-close reference.
#
# Versioned in-tree as of the P0 operational recovery, with one behavioural
# change: **a no-op sweep no longer reports health.** A sweep with nothing
# pending is a legitimate no-op and exits 0 — which is precisely how this script
# printed "0 residual orders submitted / EXIT 0" on four consecutive mornings
# (2026-07-24..29) while every nightly cycle was dying. The sweep can only
# observe its own work; it cannot observe the absence of the evening's. So it now
# asks the doctor, offline, whether the book is alive at all, and goes red if it
# is not.
set -u

REPO="${PRISM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Single switch shared with the nightly wrapper: runs/ACTIVE_RUN_DIR names the
# live run-dir (one line, relative to the repo).
RUN_DIR="$(cat "$REPO/runs/ACTIVE_RUN_DIR" 2>/dev/null || echo runs/paper_loop_momentum)"
LOG_DIR="$REPO/$RUN_DIR"
LOG="$LOG_DIR/sweep.log"
mkdir -p "$LOG_DIR"
ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }
# shellcheck source=ops/_alert.sh
. "$REPO/ops/_alert.sh"

set -a
# shellcheck disable=SC1091
[ -f "$REPO/.env" ] && . "$REPO/.env"
set +a

if [ -z "${APCA_API_KEY_ID:-}" ] || [ -z "${APCA_API_SECRET_KEY:-}" ]; then
    echo "$(ts) SKIP: APCA credentials not present in $REPO/.env" >>"$LOG"
    exit 3
fi

cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python3"
echo "$(ts) SWEEP: run-dir=$RUN_DIR" >>"$LOG"
"$PY" -m prism.scripts.paper_sweep \
    --run-dir "$RUN_DIR" \
    >>"$LOG" 2>&1
rc=$?
echo "$(ts) EXIT $rc" >>"$LOG"

# Offline only: the health question here is "did last night happen?", which is
# answered from the durable record in the run directory (equity ledger, nightly
# log, regime ledger). Keeping it offline also keeps the morning path free of
# vendor calls and of any dependency on venue reachability.
echo "$(ts) HEALTH:" >>"$LOG"
"$PY" -m prism.scripts.doctor --run-dir "$RUN_DIR" >>"$LOG" 2>&1
health_rc=$?
echo "$(ts) HEALTH EXIT $health_rc" >>"$LOG"

if [ "$rc" -ne 0 ] || [ "$health_rc" -ne 0 ]; then
    alert "$LOG" "$LOG_DIR" \
        "prism morning health red (sweep=$rc doctor=$health_rc)" \
        "run-dir $RUN_DIR; a green sweep is not evidence of a live book. Read the FAIL lines in $LOG."
fi
# Deliberately no clear_alert here: the nightly owns the healthy path. A morning
# that looks fine must never retire an alert raised by the session that trades.

[ "$rc" -ne 0 ] && exit "$rc"
exit "$health_rc"
