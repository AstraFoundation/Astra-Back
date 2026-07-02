#!/usr/bin/env bash
# Prove the astra-ai-sdk pip package works "from elsewhere":
#   wheel build → fresh venv in /tmp → pip install the wheel → run from /tmp
#   (repo can't shadow the import) → local serving + telemetry → assert the
#   dashboard saw everything (client events, hosts, breakdown, drift alert).
#
# Usage: scripts/verify_sdk_e2e.sh [BASE_URL]   (default http://localhost:8100)
set -euo pipefail

BASE_URL="${1:-http://localhost:8100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST=/tmp/astra-ai-sdk-dist
VENV=/tmp/astra-ai-sdk-venv
WORK=/tmp/astra-ai-sdk-e2e

echo "── 1. build wheel"
rm -rf "$DIST" && python3 -m build --wheel --outdir "$DIST" "$ROOT/clients/python" > /dev/null
WHEEL="$(ls "$DIST"/astra_ai_sdk-*.whl)"
echo "   $WHEEL"

echo "── 2. fresh venv + install from wheel ((onnxruntime included by default) extra)"
rm -rf "$VENV" && python3 -m venv "$VENV"
"$VENV/bin/pip" install -q "${WHEEL}"
"$VENV/bin/python" -c "import astra_sdk; print('   astra-ai-sdk', astra_sdk.__version__)"

echo "── 3. provision a deployment on the backend ($BASE_URL)"
rm -rf "$WORK" && mkdir -p "$WORK"
python3 "$ROOT/scripts/_sdk_e2e_provision.py" --base "$BASE_URL" --out "$WORK/handoff.json"

echo "── 4. serve locally from /tmp with the installed wheel"
(cd /tmp && ASTRA_SDK_WINDOW_MAX_REQUESTS=20 ASTRA_SDK_FLUSH_INTERVAL_S=1 \
    ASTRA_SDK_SNAPSHOT_INTERVAL_S=3 \
    "$VENV/bin/python" "$ROOT/scripts/_sdk_e2e_client.py" --handoff "$WORK/handoff.json")

echo "── 5. assert the dashboard saw it"
python3 "$ROOT/scripts/_sdk_e2e_assert.py" --base "$BASE_URL" --handoff "$WORK/handoff.json"

echo
echo "SDK E2E VERIFIED — installed from wheel in a fresh venv, served locally, monitored remotely"
