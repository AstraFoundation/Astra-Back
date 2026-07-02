# astra-ai-sdk (Node)

Serve **Astra-compressed models** on your own hardware — Astra never runs your
model server-side — and keep the Astra dashboard monitoring them while they run.
The Node client mirrors the [Python `astra-ai-sdk`](../python) API.

```bash
npm i astra-ai-sdk onnxruntime-node   # on-device ONNX serving on your hardware
```

Requires **Node ≥ 18.17** (global `fetch`). `onnxruntime-node` is an optional
dependency — install it to use `AstraRunner` / `astra serve`.

Exports: `AstraRunner`, `AstraRunnerError`, `pullArtifact`, `AstraTelemetryReporter`,
`telemetryEnabled`, `AstraApiError`, `VERSION`.

## On-device serving (the whole product)

`AstraRunner.fromDeployment` pulls the deployed, compressed artifact once
(sha256-cached under `~/.cache/astra`) and runs it with onnxruntime **inside
your own code** — no server to stand up, no hosted inference:

```ts
import { AstraRunner } from "astra-ai-sdk";

// baseUrl defaults to the hosted Astra origin (override with ASTRA_BASE_URL).
const runner = await AstraRunner.fromDeployment({
  deploymentId: "dep_ab12cd34ef",
  apiKey: "astra_sk_live_...",
});

// Inputs are name → { data, dims, type? }. type defaults to "float32".
const out = await runner.run({
  input: { data: new Float32Array(1 * 128), dims: [1, 128] },
});
console.log(out.latencyMs, out.outputs);

await runner.close(); // final telemetry flush
```

### Run a file you already have

Downloaded the artifact (SDK Hub → **Download Artifact**) or have an `.onnx` on
disk? Skip the deployment — serve the file directly:

```ts
import { AstraRunner } from "astra-ai-sdk";

const runner = await AstraRunner.fromFile("compressed.onnx");
const out = await runner.run({ input: { data: myFloats, dims: [1, 3, 224, 224] } });
await runner.close();
```

Telemetry is off for a bare file; pass `{ deploymentId, apiKey }` to still report
local runs to that deployment.

## Closed-loop telemetry (offline-durable)

Every on-device inference is measured and shipped back to Astra through a
**durable closed loop** that can never block or break your serving path:

- buffers events in memory and **spools them to disk when offline**;
- **flushes** the buffered batches automatically on reconnect;
- **deletes each batch only after the server acks it**, with per-batch
  idempotency so a retry is never double-counted.

| Stream | Cadence | Fields |
|---|---|---|
| **Request events** | per inference | timestamp, latency breakdown (pre / inference / post ms), success / error code, batch size, region tag, input shape signature |
| **System snapshots** | ~30 s | CPU %, RSS MB, throughput req/min, dropped-event count, SDK / onnxruntime versions, OS, arch, execution provider, hostname |
| **Window stats** | ~60 s or 200 requests | per-input tensor mean/std/min/max/NaN%, output class distribution (top-10), 16-bin confidence histogram, mean entropy, mean top-1 confidence |

Window stats power the dashboard's **prediction drift** and **input
distribution shift** alerts.

Opt out any time: `AstraRunner.fromDeployment({ ..., reportTelemetry: false })`
or `ASTRA_SDK_TELEMETRY=0`. Disable the on-disk spool with `ASTRA_SDK_SPOOL=0`
(telemetry then buffers in memory only).

## CLI

```bash
astra pull                  # pull the compressed artifact
astra serve --port 8765     # on-device HTTP endpoint: POST /infer
astra bench -n 200          # on-device p50/p95, reported as telemetry
```

`astra serve` is an optional zero-code convenience that wraps `AstraRunner` in a
`POST /infer` endpoint on `127.0.0.1`. Options can also come from
`ASTRA_BASE_URL`, `ASTRA_DEPLOYMENT_ID`, `ASTRA_API_KEY`.

## Supported platforms

`onnxruntime-node` ships prebuilt binaries for common platforms (macOS/Linux/
Windows on x64/arm64). On platforms it doesn't cover, `npm i astra-ai-sdk` still
succeeds (it's an optional dependency) — `astra pull` and telemetry work, and
`AstraRunner` raises a clear error until `onnxruntime-node` is available.
