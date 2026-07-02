# astra-ai-sdk

Serve **Astra-compressed models** on your own hardware — Astra never runs your
model server-side — and keep the Astra dashboard monitoring them while they run.

```bash
pip install 'astra-ai-sdk[serve]'         # on-device ONNX serving (onnxruntime, numpy)
pip install 'astra-ai-sdk[serve,system]'  # + precise CPU/RSS metrics (psutil)
```

Exports: `AstraRunner`, `AstraRunnerError`, `pull_artifact`, `AstraTelemetryReporter`,
`AstraApiError`.

## On-device serving (the whole product)

`AstraRunner.from_deployment` pulls the deployed, compressed artifact once
(sha256-cached under `~/.cache/astra`) and serves it with onnxruntime **inside
your own process** — no server to stand up, no hosted inference:

```python
from astra_sdk import AstraRunner

# base_url defaults to the hosted Astra origin (override with ASTRA_BASE_URL).
runner = AstraRunner.from_deployment("dep_ab12cd34ef", "astra_sk_live_...")
out = runner.run({"input": my_numpy_array})   # bare ndarray per input name
print(out["latencyMs"], out["outputs"], out["preMs"], out["postMs"])
runner.close()                                 # final telemetry flush
```

`run()` returns `{latencyMs, outputs, preMs, postMs, raw}`.

### Run a file you already have

Downloaded the artifact (SDK Hub → **Download Artifact**) or have an `.onnx` on
disk? Skip the deployment — serve the file directly:

```python
from astra_sdk import AstraRunner

runner = AstraRunner.from_file("compressed.onnx")
out = runner.run({"input": my_numpy_array})
runner.close()
```

Telemetry is off for a bare file; pass `deployment_id=` + `api_key=` to still
report local runs to that deployment.

## Closed-loop telemetry (offline-durable)

Every on-device inference is measured and shipped back to Astra through a
**durable closed loop** that can never block or break your serving path:

- buffers events in memory and **spools them to disk when offline**;
- **flushes** the buffered batches automatically on reconnect;
- **deletes each batch only after the server acks it**, with per-batch
  idempotency so a retry is never double-counted.

| Stream | Cadence | Fields |
|---|---|---|
| **Request events** | per inference | timestamp, latency breakdown (preprocess / inference / postprocess ms), success / error code, batch size, region tag, input shape signature |
| **System snapshots** | ~30 s | CPU %, RSS MB, throughput req/min, dropped-event count, SDK / Python / onnxruntime versions, OS, arch, execution provider, hostname |
| **Window stats** | ~60 s or 200 requests | per-input tensor mean/std/min/max/NaN%, output class distribution (top-10), 16-bin confidence histogram, mean entropy, mean top-1 confidence |

Window stats power the dashboard's **prediction drift** (PSI vs the
deployment's reference distribution) and **input distribution shift** alerts.

Opt out any time: `AstraRunner.from_deployment(..., report_telemetry=False)`
or `ASTRA_SDK_TELEMETRY=0`. Disable the on-disk spool with `ASTRA_SDK_SPOOL=0`
(telemetry then buffers in memory only).

## CLI

```bash
astra pull                  # pull the compressed artifact
astra serve --port 8765     # on-device HTTP endpoint: POST /infer
astra bench -n 200          # on-device p50/p95, reported as telemetry
```

Options can come from `ASTRA_BASE_URL`, `ASTRA_DEPLOYMENT_ID`, `ASTRA_API_KEY`
(or `--deployment` / `--api-key` / `--base-url` flags).
