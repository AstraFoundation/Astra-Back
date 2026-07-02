#!/usr/bin/env bash
# Real, end-to-end proof of the HARDWARE-AWARE telemetry closed loop:
#
#   boot backend → build wheel → fresh venv (repo can't shadow the import) →
#   deploy a model → `astra serve` it on a SEPARATE Python HTTP server (:8765) →
#   drive REAL HTTP inference at that server → assert the dashboard shows live
#   per-hardware speed + resource utilization with genuine hardware identity →
#   inject a multi-accelerator fleet → assert the GPU views light up.
#
# Self-contained: starts its own backend + serve process and tears them down.
# Usage: scripts/verify_hardware_telemetry.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8122
SERVE_PORT=8765
BASE_URL="http://127.0.0.1:${PORT}"
DIST=/tmp/astra-hw-dist
VENV=/tmp/astra-hw-venv
WORK=/tmp/astra-hw-e2e
RUNDIR=/tmp/astra-hw-run

BACKEND_PID=""
SERVE_PID=""
cleanup() {
  [ -n "$SERVE_PID" ] && kill -INT "$SERVE_PID" 2>/dev/null || true
  sleep 1
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null || true
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "── 1. boot backend ($BASE_URL) with sim + inline monitor + fast pipeline"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
# `exec` so $! is the real python PID (not a wrapping subshell) — the cleanup
# trap then signals uvicorn/astra directly instead of orphaning their children.
( cd "$ROOT" && exec env \
  ASTRA_DB_PATH="$RUNDIR/db.sqlite" ASTRA_STORAGE_DIR="$RUNDIR/storage" \
  ASTRA_WORK_DIR="$RUNDIR/work" ASTRA_FAST_PIPELINE=1 ASTRA_INLINE_JOBS=1 \
  ASTRA_MONITOR_INLINE_ENABLED=1 ASTRA_MONITOR_INTERVAL_SEC=5 \
  ASTRA_TELEMETRY_SIM_ENABLED=1 ASTRA_COOKIE_SECURE=0 ASTRA_RATE_LIMIT_ENABLED=0 \
  python3 -m uvicorn app.main:app --port "$PORT" --log-level warning ) \
  > "$RUNDIR/backend.log" 2>&1 &
BACKEND_PID=$!

for _ in $(seq 1 60); do
  curl -fsS "$BASE_URL/healthz" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS "$BASE_URL/healthz" >/dev/null || { echo "backend never became healthy"; cat "$RUNDIR/backend.log"; exit 1; }
echo "   backend up (pid $BACKEND_PID)"

echo "── 2. build wheel + fresh venv (onnxruntime included by default)"
rm -rf "$DIST" && python3 -m build --wheel --outdir "$DIST" "$ROOT/clients/python" > /dev/null
WHEEL="$(ls "$DIST"/astra_ai_sdk-*.whl)"
rm -rf "$VENV" && python3 -m venv "$VENV"
"$VENV/bin/pip" install -q "${WHEEL}"
echo "   $("$VENV/bin/python" -c 'import astra_sdk,sys; print("astra-ai-sdk", astra_sdk.__version__, "from", astra_sdk.__file__)')"

echo "── 3. provision a deployment"
rm -rf "$WORK" && mkdir -p "$WORK"
python3 "$ROOT/scripts/_sdk_e2e_provision.py" --base "$BASE_URL" --out "$WORK/handoff.json"
DEP=$(python3 -c "import json;print(json.load(open('$WORK/handoff.json'))['deploymentId'])")
KEY=$(python3 -c "import json;print(json.load(open('$WORK/handoff.json'))['apiKey'])")

echo "── 4. serve it on a SEPARATE Python HTTP server (:$SERVE_PORT) from the venv"
( cd /tmp && exec env ASTRA_SDK_SNAPSHOT_INTERVAL_S=2 ASTRA_SDK_FLUSH_INTERVAL_S=1 \
    ASTRA_SDK_WINDOW_MAX_REQUESTS=20 \
    "$VENV/bin/astra" serve --base-url "$BASE_URL" --deployment "$DEP" \
    --api-key "$KEY" --port "$SERVE_PORT" ) > "$RUNDIR/serve.log" 2>&1 &
SERVE_PID=$!
echo "   serve pid $SERVE_PID — http://127.0.0.1:$SERVE_PORT/infer"

echo "── 5. drive REAL HTTP inference at the separate server"
python3 "$ROOT/scripts/_hw_e2e_drive.py" --url "http://127.0.0.1:$SERVE_PORT/infer" -n 150

echo "── 6. stop serve (SIGINT → final telemetry flush) and let it land"
kill -INT "$SERVE_PID" 2>/dev/null || true
sleep 3
SERVE_PID=""

echo "── 7. assert the dashboard (real hardware fields + fleet GPU views)"
python3 "$ROOT/scripts/_hw_e2e_assert.py" --base "$BASE_URL" --handoff "$WORK/handoff.json"
