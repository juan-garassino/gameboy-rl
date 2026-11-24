#!/usr/bin/env bash
# The pod acting on itself through the RunPod GraphQL API.
#   pod_self.sh cost    -> prints this pod's USD/hour (0 when unknown)
#   pod_self.sh remove  -> terminates this pod (stops billing); no-op outside RunPod
set -uo pipefail

CMD="${1:?usage: pod_self.sh cost|remove}"
POD_ID="${RUNPOD_POD_ID:-}"
API_KEY="${RUNPOD_API_KEY:-${GBRL_RUNPOD_API_KEY:-}}"

if [ -z "$POD_ID" ] || [ -z "$API_KEY" ]; then
    echo "pod_self: not on RunPod (no RUNPOD_POD_ID/API key), skipping '$CMD'" >&2
    [ "$CMD" = cost ] && echo 0
    exit 0
fi

gql() {
    curl -s --max-time 30 -X POST https://api.runpod.io/graphql -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" -d "$1"
}

case "$CMD" in
    cost)
        gql "{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { costPerHr } }\"}" \
            | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["data"]["pod"]["costPerHr"] or 0)
except Exception: print(0)'
        ;;
    remove)
        # Billing only stops if this succeeds: retry with backoff; a GraphQL "errors" field is a failure too.
        backoff="${POD_SELF_BACKOFF:-2}"
        for attempt in 1 2 3 4 5; do
            echo "pod_self: terminating pod ${POD_ID} (attempt ${attempt})" >&2
            resp="$(gql "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}")"
            if [ -n "$resp" ] && ! printf '%s' "$resp" | grep -q '"errors"'; then
                exit 0
            fi
            echo "pod_self: terminate failed: ${resp:-no response}" >&2
            [ "$attempt" -lt 5 ] && sleep $((backoff * attempt * attempt))
        done
        echo "pod_self: could not terminate pod ${POD_ID} — STOP IT BY HAND" >&2
        exit 1
        ;;
    *)
        echo "unknown command: $CMD" >&2
        exit 2
        ;;
esac
