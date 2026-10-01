#!/usr/bin/env bash
# Off-box backup of prism's irreplaceable local artifacts: runs/ (the paper
# ledgers, which can only be re-accumulated, never regenerated), logs/, and the
# untracked results/ tree, mirrored into a dedicated git repository and pushed
# to a private remote. Consequence of the 2026-07-18 disk incident
# (docs/security.md par.3). Excludes .env deliberately.
#
# Versioned in-tree as of 2026-09-30, replacing ~/bin/prism_artifacts_sync.sh.
# That script had two defects that together hid six weeks without a backup:
#
#   1. It mirrored results/ whole. GitHub's pre-receive hook refuses any blob
#      over 100 MB (GH001), and results/run-*/record.json files had grown past
#      it, so every push since the loop restarted on 2026-09-25 was rejected.
#      The last push that landed was 2026-08-14.
#   2. It logged PUSH-FAILED and exited 0, and the nightly wrapper ignored its
#      exit status anyway. Nothing downstream could tell a failed backup from
#      a good one.
#
# This version mirrors runs/ and logs/ exactly, mirrors results/ only up to
# PRISM_SYNC_MAX_MB per file and ships a gzip copy of anything larger (a
# compressed record.json is a few MB), reports every stage to a status file the
# health check reads, and exits nonzero on any failure so the wrapper can alert
# and fold the failure into the session's exit code.
#
# Environment:
#   PRISM_REPO             the source checkout (default: the parent of ops/)
#   PRISM_ARTIFACTS_REPO   the backup clone (default: $HOME/prism-artifacts)
#   PRISM_SYNC_MAX_MB      per-file ceiling for raw results/ copies (default 45:
#                          under GitHub's 50 MB warning, well under its 100 MB
#                          refusal)
#   PRISM_SYNC_LOG         log file (default: <active run dir>/sync.log)
#
# Status file: <active run dir>/sync_status.json in the SOURCE repo, where the
# health check (prism.scripts.doctor, artifact-sync row) reads it; runs/ falls
# back to runs/sync_status.json when runs/ACTIVE_RUN_DIR is missing. The two
# bookkeeping files are excluded from the mirror so the backup's content is a
# pure function of the source and "no change" means no change.
#
# Oversize blobs already committed locally by the old script are NOT rewritten
# here: history surgery on a backup repo is the operator's act. When a push is
# refused for that reason the status and the log say so explicitly, and
# docs/operations.md (2026-09-30) gives the one-time repair.
set -u

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="${PRISM_REPO:-$(cd "$(dirname "$SELF")/.." && pwd)}"
BK="${PRISM_ARTIFACTS_REPO:-$HOME/prism-artifacts}"
MAX_MB="${PRISM_SYNC_MAX_MB:-45}"
RUN_DIR="$(cat "$REPO/runs/ACTIVE_RUN_DIR" 2>/dev/null || true)"
# The pointer is relative to the repo root or absolute (prism.live.safety.require_active_run_dir).
case "$RUN_DIR" in
    '')  RUN_PATH="$REPO/runs" ;;
    /*)  RUN_PATH="$RUN_DIR" ;;
    *)   RUN_PATH="$REPO/$RUN_DIR" ;;
esac
STATUS="$RUN_PATH/sync_status.json"
LOG="${PRISM_SYNC_LOG:-$RUN_PATH/sync.log}"
mkdir -p "$(dirname "$STATUS")" "$(dirname "$LOG")"

ts() { date -u "+%Y-%m-%dT%H:%M:%S+00:00"; }
log() { echo "$(ts) $*" >>"$LOG"; }

PUSHED=false
COMPRESSED=0
RESULTS_ERR=""   # first results/-stage error; reported after runs/ and logs/ are committed and pushed

# One JSON string: control bytes (0x00-0x1F) stripped, backslashes and quotes escaped.
json_str() { printf '%s' "$1" | LC_ALL=C tr -d '\000-\037' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# write_status <ok> <stage> <detail> — atomic: temp file then rename.
write_status() {
    local ok="$1" stage="$2" detail="$3" head tmp
    head="$(git -C "$BK" rev-parse --short HEAD 2>/dev/null || true)"
    if [ -n "$head" ]; then head="\"$head\""; else head=null; fi
    tmp="$STATUS.tmp.$$"
    printf '{"at": "%s", "ok": %s, "stage": "%s", "detail": "%s", "head": %s, "pushed": %s, "compressed": %d}\n' \
        "$(ts)" "$ok" "$stage" "$(json_str "$detail")" "$head" "$PUSHED" "$COMPRESSED" >"$tmp" \
        && mv -f "$tmp" "$STATUS"
}

# fail <stage> <detail> — log, record, exit nonzero. Never exit 0 on a failure:
# the old script's exit 0 is what made six weeks of PUSH-FAILED invisible.
fail() {
    log "FAIL [$1] $2"
    write_status false "$1" "$2"
    exit 1
}

case "$MAX_MB" in
    ''|*[!0-9]*) fail repo "PRISM_SYNC_MAX_MB=$MAX_MB is not a whole number of megabytes" ;;
esac
MAX_BYTES=$((MAX_MB * 1024 * 1024))

# --- repo -------------------------------------------------------------------
[ -d "$REPO/runs" ] || fail repo "$REPO/runs does not exist; PRISM_REPO=$REPO is not the prism checkout"
[ -d "$REPO/results" ] || fail repo "$REPO/results does not exist; nothing to back up from PRISM_REPO=$REPO"
git -C "$BK" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail repo "$BK is not a git work tree; clone the backup remote there (docs/security.md par.3)"
log "SYNC: source=$REPO backup=$BK max=${MAX_MB}MB script=$SELF"

# --- rsync ------------------------------------------------------------------
# runs/ and logs/ are mirrored exactly, deletions included: a retired run
# directory is recoverable from the backup repo's history, not its worktree
# (docs/operations.md). The sync's own two files stay out of the mirror.
rsync -a --delete --exclude=sync_status.json --exclude=sync.log "$REPO/runs/" "$BK/runs/" 2>>"$LOG"
rc=$?
[ "$rc" -eq 0 ] || fail rsync "rsync runs/ exited $rc"
if [ -d "$REPO/logs" ]; then
    rsync -a --delete "$REPO/logs/" "$BK/logs/" 2>>"$LOG"
    rc=$?
    [ "$rc" -eq 0 ] || fail rsync "rsync logs/ exited $rc"
else
    log "logs/ absent at source; skipped"
fi
# results/ is additive (no --delete, as before) and capped per file; exit 24
# (a source file vanished mid-transfer) is tolerated because research jobs
# write there while the nightly runs, and the next sync picks the file up.
# Any other error is recorded and reported AFTER the commit and push: runs/
# and logs/ have already been mirrored into the worktree by now, and the
# ledgers must not stay unpushed because of a problem in the results/ tree.
rsync -a --max-size="${MAX_MB}m" "$REPO/results/" "$BK/results/" 2>>"$LOG"
rc=$?
case "$rc" in
    0) ;;
    24) log "rsync results/ exited 24 (source files vanished during transfer); continuing" ;;
    *) RESULTS_ERR="rsync results/ exited $rc"
       log "ERROR [rsync] $RESULTS_ERR; continuing so runs/ and logs/ still commit and push; this sync will exit nonzero" ;;
esac

# Oversize results files travel compressed. gzip -n leaves name and mtime out
# of the header, so an unchanged file compresses to identical bytes and the
# backup commit stays empty. Skip when the existing .gz is newer than the
# source; write to a temp name and rename so a crash never leaves a truncated
# archive under the final name. A gzip failure is deferred like an rsync one.
#
# drop_raw: the source is over the cap, so any raw copy of it under the backup
# is from a smaller earlier version that rsync --max-size no longer refreshes;
# left alone it would sit at the canonical path beside a current .gz forever.
# Reads $raw and $rel from the loop, and runs in both branches of the loop
# because a stale raw beside an already-current .gz takes the skip branch.
drop_raw() {
    [ -f "$raw" ] || return 0
    rm -f -- "$raw" && log "DROPPED stale raw copy results/$rel (source grew past ${MAX_MB}MB; current as .gz)"
}
while IFS= read -r -d '' src; do
    rel="${src#"$REPO/results/"}"
    dst="$BK/results/$rel.gz"
    raw="$BK/results/$rel"
    if [ -f "$dst" ] && [ "$dst" -nt "$src" ]; then
        drop_raw
        continue
    fi
    mkdir -p "$(dirname "$dst")" || {
        RESULTS_ERR="${RESULTS_ERR:-cannot create $(dirname "$dst")}"
        log "ERROR [rsync] cannot create $(dirname "$dst"); continuing"
        continue
    }
    tmp="$dst.tmp"
    if gzip -n -c "$src" >"$tmp" && mv -f "$tmp" "$dst"; then
        COMPRESSED=$((COMPRESSED + 1))
        log "COMPRESSED results/$rel -> results/$rel.gz"
        drop_raw
    else
        rm -f "$tmp"
        RESULTS_ERR="${RESULTS_ERR:-gzip of results/$rel failed}"
        log "ERROR [rsync] gzip of results/$rel failed; continuing"
    fi
done < <(find "$REPO/results" -type f -size +"${MAX_BYTES}"c -print0)

# Raw oversize copies under the backup's results/ are what the old script put
# there and what the remote refuses; each is derivable from the source (and
# now present as .gz), so drop them from the worktree.
while IFS= read -r -d '' raw; do
    rm -f -- "$raw" && log "DROPPED oversize raw copy ${raw#"$BK/"} (> ${MAX_MB}MB; kept as .gz)"
done < <(find "$BK/results" -type f -size +"${MAX_BYTES}"c ! -name '*.gz' -print0 2>/dev/null)

# --- commit -----------------------------------------------------------------
git -C "$BK" add -A >>"$LOG" 2>&1 || fail commit "git add -A failed"
if git -C "$BK" diff --cached --quiet 2>>"$LOG"; then
    CHANGED=0
    log "NO-CHANGE"
else
    CHANGED=1
    git -C "$BK" commit -q -m "backup: $(date -u +%FT%H:%MZ) nightly sync" >>"$LOG" 2>&1 \
        || fail commit "git commit failed (no identity configured, or a hook refused)"
    log "COMMITTED $(git -C "$BK" rev-parse --short HEAD)"
fi

# --- push -------------------------------------------------------------------
# Always push, even with no new commit: an earlier refused push leaves local
# commits ahead of the remote, and a quiet night is when they should land.
PUSH_ERR="$(mktemp "${TMPDIR:-/tmp}/prism_sync_push.XXXXXX")"
if git -C "$BK" push -q origin HEAD:main 2>"$PUSH_ERR"; then
    PUSHED=true
    rm -f "$PUSH_ERR"
else
    rc=$?
    cat "$PUSH_ERR" >>"$LOG"
    ahead="$(git -C "$BK" rev-list --count origin/main..HEAD 2>/dev/null || echo '?')"
    if grep -q -e GH001 -e "exceeds GitHub's file size limit" "$PUSH_ERR"; then
        # The remote's limit is 100 MiB regardless of PRISM_SYNC_MAX_MB, and only
        # results/ is capped: runs/ and logs/ mirror uncapped and .gz copies are
        # never dropped, so look at the committed tree before blaming history.
        named="$(grep -o -m1 'File [^;]* is [0-9.]* MB' "$PUSH_ERR" || true)"
        big="$(find "$BK" -path "$BK/.git" -prune -o -type f -size +100M -print 2>/dev/null | sed "s#^$BK/##" | tr '\n' ' ')"
        rm -f "$PUSH_ERR"
        if [ -n "$big" ]; then
            log "PUSH-REFUSED: the remote rejects a blob over its size limit (${named:-GH001}). The current backup worktree itself holds file(s) over 100MB: $big. runs/ and logs/ are mirrored uncapped and .gz copies are never dropped, so this is in the latest commit, not old history; reset --soft will not clear it."
            fail push "push refused by the remote's file size limit (GH001): oversize file(s) in the current tree: $big; ${named:-}"
        fi
        log "PUSH-REFUSED: the remote rejects a blob over its size limit (${named:-GH001}). This run's worktree holds no file over 100MB, so the blob is in the $ahead unpushed local commit(s) the old script left behind. A one-time operator repair of the backup repo's unpushed history is required (docs/operations.md, 2026-09-30); this script does not rewrite history."
        fail push "push refused by the remote's file size limit (GH001): $ahead unpushed local commit(s) still contain oversize blobs; ${named:-}; one-time operator repair required (docs/operations.md 2026-09-30)"
    fi
    rm -f "$PUSH_ERR"
    fail push "git push origin HEAD:main exited $rc with $ahead local commit(s) unpushed; the commit is kept locally"
fi

# --- done -------------------------------------------------------------------
head="$(git -C "$BK" rev-parse --short HEAD)"
# A deferred results/-stage error is still a failure (ok:false, stage rsync,
# nonzero exit); it is reported here so that it never blocked the ledgers.
if [ -n "$RESULTS_ERR" ]; then
    fail rsync "$RESULTS_ERR (runs/ and logs/ were committed and pushed at $head; results/ copy is incomplete)"
fi
if [ "$CHANGED" -eq 1 ]; then
    detail="committed and pushed $head; $COMPRESSED file(s) compressed"
    log "SYNCED $head"
else
    detail="no-change; remote up to date at $head"
fi
write_status true "done" "$detail"
exit 0
