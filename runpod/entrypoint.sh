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
# Every gcloud storage call is capped by GCS_TIMEOUT (300 s).
# Optional: INIT_CKPT (default poke_26214400.zip; "" = fresh policy), ROM_SHA1 ("" = skip check).
# RunPod injects RUNPOD_POD_ID.
set -uo pipefail

APP_DIR="${APP_DIR:-/app}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
POD_SELF="${APP_DIR}/runpod/scripts/pod_self.sh"
TOKEN_FILE="${WORKSPACE_DIR}/control/gcs_token"
GCS_TIMEOUT="${GCS_TIMEOUT:-300}"
RUN_DIR="" GCS_ROOT="" SYNC_PID="" SKIP_SYNC=0

log() { echo "=== $* ==="; }

gcs() {
    # Re-read the token before every call (token_writer.sh may swap it); never let a hung gcloud call block the pod.
    if [ -s "$TOKEN_FILE" ]; then
        CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$TOKEN_FILE")" timeout "$GCS_TIMEOUT" gcloud storage "$@"
    else
        timeout "$GCS_TIMEOUT" gcloud storage "$@"
    fi
}

sync_up() {
    [ -n "$RUN_DIR" ] && [ -n "$GCS_ROOT" ] || return 0
    gcs rsync --recursive "$RUN_DIR" "${GCS_ROOT}/runs/${RUN_ID}" || log "sync to GCS failed (retrying next interval)"
}

finish() {
    local rc=$?
    trap - EXIT
    [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
    if [ "$SKIP_SYNC" = 0 ]; then
        log "final sync (exit code $rc)"
        sync_up
    fi
    bash "$POD_SELF" remove
    exit "$rc"
}
# Installed before any check, so even a missing RUN_ID / GCS_BUCKET ends with the pod removed (billing stops).
trap finish EXIT

MODE="${MODE:-train}"
RUN_ID="${RUN_ID:?RUN_ID not set}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET not set}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
INIT_CKPT="${INIT_CKPT-poke_26214400.zip}"
ROM_NAME="pokemon_red_version.gb"
ROM_SHA1="${ROM_SHA1-ea9bcae617fdf159b045185467ae58b2e4a48b9a}"

GCS_ROOT="gs://${GCS_BUCKET}/gameboy-rl"
RUN_DIR="${WORKSPACE_DIR}/runs/${RUN_ID}"
ROM_LOCAL="${WORKSPACE_DIR}/roms/${ROM_NAME}"

export GBRL_RUNS_DIR="${WORKSPACE_DIR}/runs" GBRL_RUN_ID="$RUN_ID" GBRL_ROM_PATH="$ROM_LOCAL"
# Bound every MLflow HTTP call so a dead tracker can't outlast the stall watchdog.
export MLFLOW_HTTP_REQUEST_MAX_RETRIES="${MLFLOW_HTTP_REQUEST_MAX_RETRIES:-2}"
export MLFLOW_HTTP_REQUEST_TIMEOUT="${MLFLOW_HTTP_REQUEST_TIMEOUT:-20}"
mkdir -p "$RUN_DIR" "$(dirname "$TOKEN_FILE")" "${WORKSPACE_DIR}/roms" "${WORKSPACE_DIR}/init"

if [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
    umask 077
    printf '%s' "$GCS_ACCESS_TOKEN" > "${TOKEN_FILE}.new" && mv "${TOKEN_FILE}.new" "$TOKEN_FILE"
    umask 022
fi

write_health() {  # status reason
    printf '{"status": "%s", "reasons": ["%s"], "last": {}}\n' "$1" "$2" > "${RUN_DIR}/health.json"
}

if gcs stat "${GCS_ROOT}/runs/${RUN_ID}/DONE" 2>/dev/null; then
    log "run ${RUN_ID} already DONE, nothing to do"
    SKIP_SYNC=1
    exit 0
fi

log "pulling run dir (resume)"
gcs rsync --recursive "${GCS_ROOT}/runs/${RUN_ID}" "$RUN_DIR" 2>/dev/null || log "no prior state, fresh run"

if compgen -G "${RUN_DIR}/checkpoints/ckpt_*.zip" > /dev/null; then
    log "run has its own checkpoint, resuming from it"
elif [ -n "$INIT_CKPT" ]; then
    log "pulling init checkpoint ${INIT_CKPT}"
    if ! gcs cp "${GCS_ROOT}/init/${INIT_CKPT}" "${WORKSPACE_DIR}/init/${INIT_CKPT}"; then
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
    if ! gcs cp "${GCS_ROOT}/roms/${ROM_NAME}" "$ROM_LOCAL"; then
        log "no ROM in GCS and none on disk"
        write_health CRITICAL rom
        exit 1
    fi
    # ROMs are uploaded right before each launch and must not stay in GCS (Juan's rule).
    gcs rm "${GCS_ROOT}/roms/**" || log "could not delete ROMs from GCS — remove them by hand"
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
