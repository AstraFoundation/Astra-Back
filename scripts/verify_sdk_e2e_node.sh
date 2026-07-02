#!/usr/bin/env bash
# Prove the astra-ai-sdk npm package works "from elsewhere":
#   build → npm pack → fresh project in /tmp → install the tarball → run from
#   /tmp (repo can't shadow the import) → local serving + telemetry → assert the
#   dashboard saw everything (client events, hosts, breakdown, drift alert).
#
# Reuses the language-agnostic Python provision/assert helpers (they speak HTTP
# and a shared handoff.json), so this is the Node twin of verify_sdk_e2e.sh.
#
# Usage: scripts/verify_sdk_e2e_node.sh [BASE_URL]   (default http://localhost:8100)
set -euo pipefail

BASE_URL="${1:-http://localhost:8100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/clients/node"
WORK=/tmp/astra-node-e2e

echo "── 1. build + pack the npm package"
( cd "$PKG" && npm run build >/dev/null 2>&1 && npm pack --pack-destination /tmp >/dev/null )
TARBALL="$(ls -t /tmp/astra-ai-sdk-*.tgz | head -1)"
echo "   $TARBALL"

echo "── 2. fresh project in /tmp + install the tarball (onnxruntime-node auto)"
rm -rf "$WORK" /tmp/astra-node-cache && mkdir -p "$WORK"
( cd "$WORK" \
    && npm init -y >/dev/null 2>&1 \
    && npm pkg set type=module >/dev/null 2>&1 \
    && npm install -q "$TARBALL" >/dev/null 2>&1 )
node -e "console.log('   astra-ai-sdk', require('$WORK/node_modules/astra-ai-sdk/package.json').version)"
# Run the client FROM /tmp so the bare \"astra-ai-sdk\" import resolves to the
# installed tarball (ESM resolves relative to the importing file's location).
cp "$ROOT/scripts/_sdk_e2e_client.mjs" "$WORK/client.mjs"

echo "── 3. provision a deployment on the backend ($BASE_URL)"
python3 "$ROOT/scripts/_sdk_e2e_provision.py" --base "$BASE_URL" --out "$WORK/handoff.json"

echo "── 4. serve locally from /tmp with the installed package"
( cd "$WORK" && ASTRA_SDK_WINDOW_MAX_REQUESTS=20 ASTRA_SDK_FLUSH_INTERVAL_S=1 \
    ASTRA_SDK_SNAPSHOT_INTERVAL_S=3 \
    node "$WORK/client.mjs" --handoff "$WORK/handoff.json" )

echo "── 5. assert the dashboard saw it"
# Wait a full inline-monitor cycle (monitor_interval_sec defaults to 60) so the
# input-drift pass reliably runs after the shifted windows land.
python3 "$ROOT/scripts/_sdk_e2e_assert.py" --base "$BASE_URL" --handoff "$WORK/handoff.json" \
  --monitor-wait 90

echo
echo "NODE SDK E2E VERIFIED — installed from tarball in a fresh project, served locally, monitored remotely"
