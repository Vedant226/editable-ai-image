#!/usr/bin/env bash
# Bring up the ENTIRE stack in dependency order, health-gated at each step:
#
#   ComfyUI (:8188) -> wait healthy -> bridge (:8189) -> lift (:8000) -> frontend (:5173)
#
# Resilient startup: a downstream service is not started until the one it depends
# on answers its health probe, so the editor never opens against a half-up
# pipeline (the cause of "ComfyUI is offline"). Idempotent — anything already
# running is detected and left alone, so it is safe to re-run.
#
# Ctrl-C stops every service this script started. Logs go to $LOGDIR.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
LOGDIR="${LOGDIR:-/tmp/editable-ai-logs}"; mkdir -p "$LOGDIR"
pids=()

cleanup() {
  echo
  echo "[run_all] shutting down services started here…"
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

wait_health() { # name url timeout_seconds
  local name="$1" url="$2" timeout="${3:-120}" i=0
  printf "[run_all] waiting for %s %s " "$name" "$url"
  while (( i < timeout )); do
    if curl -s -m 2 -o /dev/null "$url"; then echo "OK"; return 0; fi
    printf "."; sleep 1; ((i++))
  done
  echo " TIMEOUT after ${timeout}s"
  return 1
}

ensure() { # name health_url start_cmd logfile timeout
  local name="$1" url="$2" cmd="$3" log="$4" timeout="${5:-120}"
  if curl -s -m 2 -o /dev/null "$url"; then
    echo "[run_all] $name already running — leaving it as is"
    return 0
  fi
  echo "[run_all] starting $name … (log: $log)"
  eval "$cmd" > "$log" 2>&1 &
  pids+=("$!")
  wait_health "$name" "$url" "$timeout"
}

# 1) ComfyUI first — everything AI depends on it. Give it the longest window.
ensure "ComfyUI"  "http://127.0.0.1:8188/system_stats" "./run_comfyui.sh" \
       "$LOGDIR/comfyui.log" "${COMFY_TIMEOUT:-240}" \
  || { echo "[run_all] ERROR: ComfyUI did not come up — see $LOGDIR/comfyui.log"; exit 1; }

# 2) Bridge — depends on ComfyUI; gates AI generation behind it.
ensure "bridge"   "http://127.0.0.1:8189/health" "./run_comfy_bridge.sh" \
       "$LOGDIR/bridge.log" 30 \
  || { echo "[run_all] ERROR: bridge did not come up — see $LOGDIR/bridge.log"; exit 1; }

# 3) Lift / inpaint service — independent of ComfyUI but needed for the editor.
ensure "lift"     "http://127.0.0.1:8000/health" "./run.sh" \
       "$LOGDIR/lift.log" 90 \
  || { echo "[run_all] ERROR: lift service did not come up — see $LOGDIR/lift.log"; exit 1; }

# 4) End-to-end check: does the bridge actually reach ComfyUI?
if curl -s -m 5 http://127.0.0.1:8189/comfyui/status | grep -q '"online": *true'; then
  echo "[run_all] ✓ bridge <-> ComfyUI confirmed ONLINE"
else
  echo "[run_all] ⚠ bridge is up but cannot reach ComfyUI — check $LOGDIR/comfyui.log"
fi

# 5) Frontend (Vite default :5173). Optional — skip with NO_FRONTEND=1.
if [ "${NO_FRONTEND:-0}" != "1" ]; then
  ensure "frontend" "http://127.0.0.1:5173/" "cd '$ROOT/editable-editor' && npm run dev" \
         "$LOGDIR/frontend.log" 60 || true
fi

echo
echo "[run_all] stack is up:"
echo "    ComfyUI    http://127.0.0.1:8188"
echo "    bridge     http://127.0.0.1:8189/health   (status: /comfyui/status)"
echo "    lift       http://127.0.0.1:8000/health"
echo "    frontend   http://127.0.0.1:5173"
echo "[run_all] logs: $LOGDIR   |   Ctrl-C stops services started here."

# Keep the script alive so Ctrl-C can clean up the services it launched.
wait
