#!/usr/bin/env bash
# Minimal alert transport for the PRISM_ALERT_CMD seam in ops/_alert.sh: POST
# the alert body to an ntfy-compatible endpoint with the subject as the Title
# header. Invoked by the wrappers as
#   "$PRISM_ALERT_CMD" "<subject>" "<body>"
#
# Setup (operator's act): create a topic on an ntfy server (ntfy.sh, or a
# self-hosted one; the topic name is the only secret, so make it unguessable),
# then put in $REPO/.env:
#   PRISM_ALERT_CMD=/home/b90/prism/ops/alert_curl.sh
#   PRISM_ALERT_URL=https://ntfy.sh/<topic>
# and subscribe to the topic from a phone or desktop client.
#
# Exits nonzero when PRISM_ALERT_URL is unset or curl fails, so the caller logs
# ALERT-TRANSPORT-FAILED; the caller never fails the session over that. The
# URL is never printed: like a heartbeat URL, whoever holds it can post to it.
set -u

subject="${1:?usage: alert_curl.sh <subject> <body>}"
body="${2:?usage: alert_curl.sh <subject> <body>}"

if [ -z "${PRISM_ALERT_URL:-}" ]; then
    echo "alert_curl.sh: PRISM_ALERT_URL is unset; set it in .env beside PRISM_ALERT_CMD" >&2
    exit 2
fi

# Headers are ASCII by protocol; ntfy reads Title verbatim. The subjects the
# wrappers send are plain ASCII, and anything that is not is left to the body.
# The body goes in on stdin: as a --data-binary argument, a body starting with
# '@' would name a file for curl to read instead. The pipeline's exit status is
# curl's, so the nonzero-on-failure contract above is unchanged.
printf '%s' "$body" | curl -fsS -m 15 --retry 2 -o /dev/null \
    -H "Title: $subject" \
    --data-binary @- \
    "$PRISM_ALERT_URL"
