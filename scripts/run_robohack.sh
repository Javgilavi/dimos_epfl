#!/usr/bin/env bash
# Launch the dimos agentic stack on the real Go2 + the workstation YOLO
# segmentation pipeline at the same time. The browser UI in browser_ui/
# is intentionally NOT started here — run that separately so the user can
# restart / hot-reload it without touching the robot.
#
# Usage:
#   scripts/run_robohack.sh                       # default (no simulation)
#   scripts/run_robohack.sh --simulation          # passthrough to dimos
#   YOLO_STREAM_URL=http://192.168.1.42:8888/frame scripts/run_robohack.sh
#
# Ctrl-C kills both children cleanly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv if present (works inside or outside it)
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Tunables (env-overridable)
ROBOT_IP="${ROBOT_IP:-192.168.123.161}"
YOLO_MODEL="${YOLO_MODEL:-yolo11s-seg.pt}"
YOLO_STREAM_URL="${YOLO_STREAM_URL:-http://192.168.123.18:8888/frame}"
YOLO_CONF="${YOLO_CONF:-0.3}"
YOLO_IMGSZ="${YOLO_IMGSZ:-480}"
ROBOHACK_CLOUD_URL="${ROBOHACK_CLOUD_URL:-http://localhost:8080}"
ROBOT_ID="${ROBOT_ID:-go2_a}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"

DIMOS_LOG="$LOG_DIR/dimos-${TS}.log"
YOLO_LOG="$LOG_DIR/yolo-${TS}.log"

PIDS=()
CLEANED=0

# Recursively send a signal to a PID and ALL of its descendants. dimos uses
# Python multiprocessing (forkserver), so its workers are children of the
# dimos process; killing only the parent leaves the workers running.
_kill_tree() {
    local sig="$1"; local root="$2"
    [[ -z "$root" ]] && return
    kill -0 "$root" 2>/dev/null || return
    local kids
    kids="$(pgrep -P "$root" 2>/dev/null || true)"
    for k in $kids; do _kill_tree "$sig" "$k"; done
    kill "-$sig" "$root" 2>/dev/null || true
}

cleanup() {
    # Re-entrancy guard — INT and EXIT both fire on Ctrl-C.
    [[ "$CLEANED" == "1" ]] && return
    CLEANED=1
    echo
    echo "[run_robohack] shutting down…"

    # 1) Polite TERM to each tracked PID + its descendants.
    for pid in "${PIDS[@]:-}"; do
        _kill_tree TERM "$pid"
    done

    # 2) Sweep up known dimos / yolo orphan patterns the worker pool spawns.
    #    These are children of the dimos process but multiprocessing.forkserver
    #    can outlive the parent if it's PID-reparented to init.
    pkill -TERM -f "dimos --robot-ip" 2>/dev/null || true
    pkill -TERM -f "workstation_yolo.py" 2>/dev/null || true
    pkill -TERM -f "from multiprocessing.forkserver import main.*python_lcm" 2>/dev/null || true
    pkill -TERM -f "from multiprocessing.resource_tracker" 2>/dev/null || true
    pkill -TERM -f "python_worker.py" 2>/dev/null || true

    # 3) Give them ~1.5s to exit cleanly.
    sleep 1.5

    # 4) Anything still alive → KILL.
    for pid in "${PIDS[@]:-}"; do
        _kill_tree KILL "$pid"
    done
    pkill -KILL -f "dimos --robot-ip" 2>/dev/null || true
    pkill -KILL -f "workstation_yolo.py" 2>/dev/null || true
    pkill -KILL -f "from multiprocessing.forkserver import main.*python_lcm" 2>/dev/null || true
    pkill -KILL -f "from multiprocessing.resource_tracker" 2>/dev/null || true
    pkill -KILL -f "python_worker.py" 2>/dev/null || true

    wait 2>/dev/null || true
    echo "[run_robohack] done. logs in $LOG_DIR"
}
# HUP catches "terminal closed", INT catches Ctrl-C, TERM is for kill.
trap cleanup INT TERM HUP EXIT

# Probe the Jetson HTTP frame server. dimos defaults to using it as the
# Go2's video source (GO2_USE_EXTERNAL_CAMERA=true) and YOLO pulls from it
# too. When it's down we (a) tell dimos to fall back to the native WebRTC
# camera, otherwise the UI camera goes dark, and (b) skip YOLO so it doesn't
# spam "connection refused" forever.
JETSON_HOST="$(echo "$YOLO_STREAM_URL" | sed -E 's#https?://([^:/]+).*#\1#')"
JETSON_PORT="$(echo "$YOLO_STREAM_URL" | sed -nE 's#https?://[^:/]+:([0-9]+).*#\1#p')"
JETSON_PORT="${JETSON_PORT:-80}"

JETSON_UP=0
if timeout 2 bash -c ">/dev/tcp/${JETSON_HOST}/${JETSON_PORT}" 2>/dev/null; then
    JETSON_UP=1
fi

# Decide camera source for dimos. User can override via env GO2_USE_EXTERNAL_CAMERA.
if [[ -z "${GO2_USE_EXTERNAL_CAMERA:-}" ]]; then
    if [[ "$JETSON_UP" == "1" ]]; then
        export GO2_USE_EXTERNAL_CAMERA=true
        echo "[run_robohack] Jetson camera up — dimos will use external camera."
    else
        export GO2_USE_EXTERNAL_CAMERA=false
        echo "[run_robohack] Jetson camera DOWN — dimos will use the Go2's native"
        echo "             WebRTC camera instead so the UI feed keeps working."
    fi
fi

# CI=1 makes dimos's system_configurator skip its interactive
# "Apply these changes now? [y/N]" prompt (clock sync etc.). Without this,
# dimos hangs on stdin and never publishes the camera over LCM.
echo "[run_robohack] dimos --robot-ip $ROBOT_IP run unitree-go2-agentic $* → $DIMOS_LOG"
echo "             GO2_USE_EXTERNAL_CAMERA=$GO2_USE_EXTERNAL_CAMERA"
# Use process substitution so $! is the dimos PID (not tee's). This is what
# lets Ctrl-C signal dimos directly; with `dimos | tee` the pipeline's $! is
# tee, and killing tee leaves dimos and 8 worker processes running.
CI=1 GO2_USE_EXTERNAL_CAMERA="$GO2_USE_EXTERNAL_CAMERA" \
    dimos --robot-ip "$ROBOT_IP" run unitree-go2-agentic "$@" \
    </dev/null > >(tee "$DIMOS_LOG") 2>&1 &
PIDS+=($!)

# Wait for dimos to actually be up. We hold YOLO back until *both*
#   (a) the dimos process is still alive
#   (b) the dimos log shows the WebRTC + video channels are live
# Otherwise YOLO would happily fetch frames from the Jetson and publish
# to LCM topics no one is subscribed to yet (your last run hit this).
DIMOS_PID="${PIDS[0]}"
DIMOS_READY=0
DIMOS_WAIT_S="${DIMOS_WAIT_S:-120}"

echo "[run_robohack] waiting up to ${DIMOS_WAIT_S}s for dimos to connect to robot…"
for ((i = 0; i < DIMOS_WAIT_S; i++)); do
    # Process died → bail out, don't start YOLO
    if ! kill -0 "$DIMOS_PID" 2>/dev/null; then
        echo "[run_robohack] dimos exited before becoming ready — see $DIMOS_LOG"
        exit 1
    fi
    # Match either path: native WebRTC OR external camera publishing /color_image.
    # The external-camera path doesn't print "Video channel: on", so we accept the
    # dimos coordinator finishing module startup as readiness in that case.
    if grep -qE "Video channel: on|Peer Connection State.*🟢 connected|color_image.*publish" "$DIMOS_LOG" 2>/dev/null; then
        DIMOS_READY=1
        break
    fi
    sleep 1
done

if [[ "$DIMOS_READY" != "1" ]]; then
    echo "[run_robohack] WARNING: dimos didn't reach ready state in ${DIMOS_WAIT_S}s."
    echo "             NOT starting YOLO. Tail of $DIMOS_LOG:"
    tail -20 "$DIMOS_LOG" 2>/dev/null | sed 's/^/                /'
    echo
    echo "[run_robohack] dimos is still running. Ctrl-C to stop."
    wait -n
    echo "[run_robohack] dimos exited; tearing down."
    exit 0
fi

echo "[run_robohack] dimos ready."

YOLO_OK=1
if [[ "$JETSON_UP" != "1" && "${FORCE_YOLO:-0}" != "1" ]]; then
    YOLO_OK=0
    echo
    echo "[run_robohack] Skipping workstation_yolo.py (Jetson HTTP camera not reachable"
    echo "             at ${JETSON_HOST}:${JETSON_PORT}). UI camera still works via"
    echo "             dimos LCM. To enable YOLO either:"
    echo "             (a) start scripts/jetson_camera_server.py on the Jetson, or"
    echo "             (b) override:  YOLO_STREAM_URL=http://<host>:<port>/frame …"
    echo "             (c) force:     FORCE_YOLO=1 scripts/run_robohack.sh"
    echo
fi

if [[ "$YOLO_OK" == "1" ]]; then
    echo "[run_robohack] workstation_yolo (--feed-dimos --headless) → $YOLO_LOG"
    echo "             stream=$YOLO_STREAM_URL  cloud=$ROBOHACK_CLOUD_URL"
    ROBOHACK_CLOUD_URL="$ROBOHACK_CLOUD_URL" \
    ROBOT_ID="$ROBOT_ID" \
    python scripts/workstation_yolo.py \
        --model "$YOLO_MODEL" \
        --stream-url "$YOLO_STREAM_URL" \
        --conf "$YOLO_CONF" \
        --imgsz "$YOLO_IMGSZ" \
        --feed-dimos \
        --headless </dev/null > >(tee "$YOLO_LOG") 2>&1 &
    PIDS+=($!)
fi

echo
echo "[run_robohack] dimos $( [[ "$YOLO_OK" == "1" ]] && echo '+ YOLO' ) running."
echo "             Ctrl-C here to stop."
echo "             Browser UI is NOT started — run it yourself:"
echo "                 cd browser_ui && uvicorn main:app --host 0.0.0.0 --port 8080 --reload"
echo

# Block on any child process; trap handles cleanup of the rest.
wait -n
echo "[run_robohack] one process exited; tearing the others down."
