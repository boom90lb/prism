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

# Physical path, for the same reasons as the nightly wrapper: symlink-safe
# REPO resolution and an honest provenance stamp.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="${PRISM_REPO:-$(cd "$(dirname "$SELF")/.." && pwd)}"
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

# ---------------------------------------------------------------------------
# THE RELEASE SWITCH — the section 7 migration row "the working tree is the
# live system" -> "a pinned installed release". ONE line, and it is the account
# owner's act, not a deployment detail: changing it changes which code trades.
#
#   checkout      the editable .venv of this working tree. What runs today, and
#                 the default until the owner switches, so nothing about this
#                 session changes by upgrading the wrapper alone.
#   <name>        $PRISM_RELEASE_ROOT/<name> — a pinned installed release
#                 (default root: $REPO/releases). Switch by editing this line
#                 to e.g. PRISM_RELEASE="${PRISM_RELEASE:-prism-0.3.4}".
#   /abs/path     that release root, wherever it lives.
#
# Both paths work; the wrapper takes the release to run and nothing else in this
# file changes with it. Review the upgrade first — `prism upgrade-diff` reports
# what the new release would decide differently over recorded sessions — and
# repoint the scheduler yourself; neither is done here.
PRISM_RELEASE="${PRISM_RELEASE:-checkout}"
PRISM_RELEASE_ROOT="${PRISM_RELEASE_ROOT:-$REPO/releases}"
case "$PRISM_RELEASE" in
    checkout) RELEASE_ROOT="$REPO"; PY="$REPO/.venv/bin/python3" ;;
    /*)       RELEASE_ROOT="$PRISM_RELEASE"; PY="$PRISM_RELEASE/bin/python3" ;;
    *)        RELEASE_ROOT="$PRISM_RELEASE_ROOT/$PRISM_RELEASE"; PY="$RELEASE_ROOT/bin/python3" ;;
esac
# Fail loud, never fall back: a wrapper that quietly reverted to the checkout
# after the owner pinned a release would trade code nobody approved.
if [ ! -x "$PY" ]; then
    echo "$(ts) FAIL: PRISM_RELEASE=$PRISM_RELEASE resolves to $PY, which is not an executable interpreter; refusing to run rather than falling back to the checkout" >>"$LOG"
    alert "$LOG" "$LOG_DIR" \
        "prism session refused (release $PRISM_RELEASE unusable)" \
        "PRISM_RELEASE=$PRISM_RELEASE resolved to $PY under root $RELEASE_ROOT. Install the release there, or set PRISM_RELEASE=checkout to run the working tree."
    exit 4
fi
# cd to $REPO for both paths: --run-dir and --universe-file are repo-relative,
# and the package lives under src/, so a repo cwd cannot shadow an installed
# release's own prism.
cd "$REPO" || exit 1
# Which engine actually ran, beside the wrapper and the commit. Never fatal:
# an unreadable version is a provenance gap, not a reason to skip a session.
ENGINE_VERSION="$("$PY" -c 'import importlib.metadata as m; print(m.version("prism-trading"))' 2>/dev/null || echo unknown)"
COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
[ -n "$(git -C "$REPO" status --porcelain -uno 2>/dev/null)" ] && COMMIT="${COMMIT}+dirty"
echo "$(ts) SWEEP: run-dir=$RUN_DIR wrapper=$SELF commit=$COMMIT release=$PRISM_RELEASE engine=$ENGINE_VERSION" >>"$LOG"
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
