#!/usr/bin/env bash
# Shared alerting for the paper-session wrappers. Sourced, not executed.
#
# The 2026-07-23..29 outage was not a detection failure — the tracebacks were in
# nightly.log the whole time — it was a *notification* failure: six consecutive
# nonzero nightly exits reached nobody, while the morning sweep exited 0 four
# times and prism-doctor reported "8 pass, 0 fail". So an alert here has to do
# three independent things, because each one alone has already failed:
#
#   1. Leave a durable marker in the run directory ({run-dir}/ALERT). runs/ is
#      rsynced off-box by prism_artifacts_sync.sh, so the marker survives the
#      box and is visible without reading a log.
#   2. Write a loud line into the session log next to the traceback.
#   3. Exit nonzero, so the scheduler's own last-result surface goes red.
#
# Optional escalation: export PRISM_ALERT_CMD in .env and it is invoked as
#   "$PRISM_ALERT_CMD" "<subject>" "<body>"
# The repo deliberately does not pick a transport (mail/ntfy/Pushover are all
# operator choices); it provides the seam and never fails the session because
# the seam failed.

# alert <log> <run_dir> <subject> <body>
alert() {
    local log="$1" run_dir="$2" subject="$3" body="$4"
    local stamp
    stamp="$(date "+%Y-%m-%dT%H:%M:%S%z")"
    printf '%s ALERT: %s — %s\n' "$stamp" "$subject" "$body" >>"$log"
    {
        printf '%s\n%s\n%s\n' "$stamp" "$subject" "$body"
    } >"$run_dir/ALERT"
    if [ -n "${PRISM_ALERT_CMD:-}" ]; then
        "$PRISM_ALERT_CMD" "$subject" "$body" >>"$log" 2>&1 \
            || printf '%s ALERT-TRANSPORT-FAILED: %s\n' "$stamp" "$PRISM_ALERT_CMD" >>"$log"
    fi
}

# clear_alert <log> <run_dir> — a healthy session retires the marker, so a
# present ALERT file always means "unresolved", never "happened once in July".
clear_alert() {
    local log="$1" run_dir="$2"
    if [ -f "$run_dir/ALERT" ]; then
        printf '%s ALERT-CLEARED\n' "$(date "+%Y-%m-%dT%H:%M:%S%z")" >>"$log"
        rm -f "$run_dir/ALERT"
    fi
}
