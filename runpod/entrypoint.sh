#!/usr/bin/env bash
# RunPod entrypoint for gameboy-rl (Pokemon Red).
#
#   MODE=train|smoke|check-start   (default train)
#
# Lifecycle: GCS auth -> exit if the run is already DONE -> pull the run dir (resume) ->
# pull the init checkpoint if the run has none -> pull the ROM, delete it from GCS, verify SHA-1 ->
# export pod cost -> periodic sync -> gameboy-rl $MODE -> final sync -> terminate this pod.
# The EXIT trap guarantees the last two steps, so billing stops whatever happens
# (finish, CRITICAL health, crash, stall exit code 3).
#
# The ROM is pulled LAST of the inputs: a misconfigured run (no init checkpoint) fails before it
# consumes the per-run ROM upload.
#
# Required env: RUN_ID, GCS_BUCKET; GCS_ACCESS_TOKEN (short-lived, 1 h — after that only MLflow works).
# Optional: INIT_CKPT (default poke_26214400.zip; "" = fresh policy), ROM_SHA1 ("" = skip check).
# RunPod injects RUNPOD_POD_ID.
set -uo pipefail

MODE="${MODE:-train}"
RUN_ID="${RUN_ID:?RUN_ID not set}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET not set}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
APP_DIR="${APP_DIR:-/app}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
INIT_CKPT="${INIT_CKPT-poke_26214400.zip}"
ROM_NAME="pokemon_red_version.gb"
ROM_SHA1="${ROM_SHA1-ea9bcae617fdf159b045185467ae58b2e4a48b9a}"

GCS_ROOT="gs://${GCS_BUCKET}/gameboy-rl"
RUN_DIR="${WORKSPACE_DIR}/runs/${RUN_ID}"
ROM_LOCAL="${WORKSPACE_DIR}/roms/${ROM_NAME}"
TOKEN_FILE="${WORKSPACE_DIR}/control/gcs_token"
POD_SELF="${APP_DIR}/runpod/scripts/pod_self.sh"

export GBRL_RUNS_DIR="${WORKSPACE_DIR}/runs" GBRL_RUN_ID="$RUN_ID" GBRL_ROM_PATH="$ROM_LOCAL"
mkdir -p "$RUN_DIR" "$(dirname "$TOKEN_FILE")" "${WORKSPACE_DIR}/roms" "${WORKSPACE_DIR}/init"

log() { echo "=== $* ==="; }

if [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
    umask 077
    printf '%s' "$GCS_ACCESS_TOKEN" > "${TOKEN_FILE}.new" && mv "${TOKEN_FILE}.new" "$TOKEN_FILE"
    umask 022
fi

gcs() {
    # Re-read the token before every call: token_writer.sh may swap it.
    if [ -s "$TOKEN_FILE" ]; then
        CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$TOKEN_FILE")" gsutil "$@"
    else
        gsutil "$@"
    fi
}

sync_up() {
    gcs -m -q rsync -r "$RUN_DIR" "${GCS_ROOT}/runs/${RUN_ID}" || log "sync to GCS failed (retrying next interval)"
}

write_health() {  # status reason
    printf '{"status": "%s", "reasons": ["%s"], "last": {}}\n' "$1" "$2" > "${RUN_DIR}/health.json"
}

SYNC_PID=""
finish() {
    local rc=$?
    trap - EXIT
    [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
    log "final sync (exit code $rc)"
    sync_up
    bash "$POD_SELF" remove
    exit "$rc"
}

if gcs -q stat "${GCS_ROOT}/runs/${RUN_ID}/DONE" 2>/dev/null; then
    log "run ${RUN_ID} already DONE, nothing to do"
    bash "$POD_SELF" remove
    exit 0
fi

trap finish EXIT

log "pulling run dir (resume)"
gcs -m -q rsync -r "${GCS_ROOT}/runs/${RUN_ID}" "$RUN_DIR" 2>/dev/null || log "no prior state, fresh run"

if compgen -G "${RUN_DIR}/checkpoints/ckpt_*.zip" > /dev/null; then
    log "run has its own checkpoint, resuming from it"
elif [ -n "$INIT_CKPT" ]; then
    log "pulling init checkpoint ${INIT_CKPT}"
    if ! gcs -q cp "${GCS_ROOT}/init/${INIT_CKPT}" "${WORKSPACE_DIR}/init/${INIT_CKPT}"; then
        log "init checkpoint missing (set INIT_CKPT= to start from a fresh policy)"
        write_health CRITICAL init_ckpt
        exit 1
    fi
    export GBRL_RESUME_FROM="${WORKSPACE_DIR}/init/${INIT_CKPT}"
fi

if [ -s "$ROM_LOCAL" ]; then
    log "ROM already on disk"
else
    log "pulling ROM"
    if ! gcs -q cp "${GCS_ROOT}/roms/${ROM_NAME}" "$ROM_LOCAL"; then
        log "no ROM in GCS and none on disk"
        write_health CRITICAL rom
        exit 1
    fi
    # ROMs are uploaded right before each launch and must not stay in GCS (Juan's rule).
    gcs -m -q rm "${GCS_ROOT}/roms/**" || log "could not delete ROMs from GCS — remove them by hand"
fi
if [ -n "$ROM_SHA1" ] && [ "$(sha1sum "$ROM_LOCAL" | cut -d' ' -f1)" != "$ROM_SHA1" ]; then
    log "ROM SHA-1 mismatch (expected ${ROM_SHA1})"
    rm -f "$ROM_LOCAL"
    write_health CRITICAL rom
    exit 1
fi

GBRL_COST_PER_HR_USD="$(bash "$POD_SELF" cost)"
export GBRL_COST_PER_HR_USD
log "pod cost ${GBRL_COST_PER_HR_USD} USD/h"

sync_loop() {
    # Kill our own sleep on TERM, otherwise it outlives the loop and keeps stdout open.
    local sleep_pid=""
    trap '[ -n "$sleep_pid" ] && kill "$sleep_pid" 2>/dev/null; exit 0' TERM
    while true; do
        sleep "$SYNC_INTERVAL" &
        sleep_pid=$!
        wait "$sleep_pid"
        sync_up
    done
}
sync_loop &
SYNC_PID=$!

log "gameboy-rl ${MODE} (run=${RUN_ID}, resume=${GBRL_RESUME_FROM:-run checkpoint or fresh})"
gameboy-rl "$MODE"
exit $?
