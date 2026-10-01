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
#      mirrored off-box by ops/artifacts_sync.sh, so the marker survives the
#      box and is visible without reading a log.
#   2. Write a loud line into the session log next to the traceback.
#   3. Exit nonzero, so the scheduler's own last-result surface goes red.
#
# Optional escalation: export PRISM_ALERT_CMD in .env and it is invoked as
#   "$PRISM_ALERT_CMD" "<subject>" "<body>"
# The repo deliberately does not pick a transport (mail/ntfy/Pushover are all
# operator choices); it provides the seam and never fails the session because
# the seam failed. ops/alert_curl.sh is a minimal ntfy-compatible transport for
# operators who want one.
#
# Heartbeat (added 2026-09-30): everything above runs inside the session, so
# none of it can report a scheduler that never fires — the 2026-08-15 →
# 2026-09-25 gap, when both scheduled tasks sat disabled and nothing said so.
# A dead-man's monitor closes that gap from outside: each wrapper ends by
# pinging a URL, the monitor expects a ping on schedule, and it alarms when the
# ping does not arrive. healthchecks.io convention: the wrapper's final exit
# code is appended to the URL, so <url>/0 reports success and <url>/<nonzero>
# reports a failure immediately rather than after the grace period.
#   PRISM_HEARTBEAT_URL        the nightly wrapper's monitor
#   PRISM_HEARTBEAT_URL_SWEEP  the morning sweep's monitor
# Unset means no ping (the doctor's alert-transport row warns about that). A
# failed ping is logged as HEARTBEAT-FAILED and never changes the session's
# exit code: the monitor's own missed-ping alarm covers a dead ping path.

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

# heartbeat <log> <rc> <url> — ping the dead-man's monitor at <url>/<rc>.
# The caller passes the URL variable's value ("${PRISM_HEARTBEAT_URL:-}" or
# "${PRISM_HEARTBEAT_URL_SWEEP:-}"); an empty value is a silent no-op. Always
# returns 0, and the URL is never written to the log (a ping URL is a secret:
# anyone holding it can keep the monitor green).
heartbeat() {
    local log="$1" rc="$2" url="${3:-}"
    local stamp curl_rc
    [ -n "$url" ] || return 0
    stamp="$(date "+%Y-%m-%dT%H:%M:%S%z")"
    curl -fsS -m 10 --retry 2 -o /dev/null "${url%/}/$rc" >>"$log" 2>&1
    curl_rc=$?
    if [ "$curl_rc" -eq 0 ]; then
        printf '%s HEARTBEAT rc=%s\n' "$stamp" "$rc" >>"$log"
    else
        printf '%s HEARTBEAT-FAILED rc=%s (curl exit %s); the monitor will alarm on the missed ping\n' \
            "$stamp" "$rc" "$curl_rc" >>"$log"
    fi
    return 0
}
