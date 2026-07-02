# Changelog

## 0.4.0

- **Breaking — public classes rebranded to the Astra namespace.** `LocalRunner`
  → **`AstraRunner`**, `RunnerError` → **`AstraRunnerError`**, `TelemetryReporter`
  → **`AstraTelemetryReporter`**, `ApiError` → **`AstraApiError`**. Update imports:
  `from astra_sdk import AstraRunner`. The import module (`astra_sdk`), the `astra`
  CLI, function/constant names, and all behavior are unchanged.

## 0.3.0

- **Breaking — hosted inference removed.** `AstraClient` and the hosted
  `infer()` path are gone; Astra never runs your model server-side. The
  `POST /api/v1/infer/{deployment_id}` endpoint no longer exists. Inference now
  happens only on-device via `LocalRunner`. The public API-key surface is
  `GET /api/v1/artifacts/{deployment_id}` (pull the compressed model) and
  `POST /api/v1/telemetry/{deployment_id}/batch` (closed-loop telemetry).
- **Added — durable, offline-buffered closed-loop telemetry.** On-device events
  buffer in memory and **spool to disk when offline**, **flush** on reconnect,
  and are **deleted only after the server acks** each batch, with per-batch
  idempotency so retries are never double-counted. Disable the disk spool with
  `ASTRA_SDK_SPOOL=0`.

## 0.2.0 — 2026-06-11

- **Local serving**: `LocalRunner.from_deployment()` pulls the deployed,
  Astra-compressed artifact (sha256/ETag-cached on disk) and serves it with
  onnxruntime on your own hardware (`pip install 'astra-ai-sdk[serve]'`).
- **Run a local file**: `LocalRunner.from_file("model.onnx")` serves an artifact
  you already have (e.g. the SDK Hub "Download Artifact" file) — no deployment
  needed; telemetry off unless you pass `deployment_id` + `api_key`.
- **Built-in telemetry**: every local inference is measured (latency
  breakdown, batch, input signature) and shipped in fault-tolerant background
  batches to the Astra dashboard — plus periodic system snapshots (CPU/RSS/
  throughput/runtime fingerprint) and windowed input/output distribution
  stats that power prediction/input drift alerts. Opt out with
  `report_telemetry=False` or `ASTRA_SDK_TELEMETRY=0`.
- **No base URL in your code**: `base_url` is optional everywhere — it defaults
  to the hosted Astra origin (override with `ASTRA_BASE_URL` or the `base_url`
  keyword). `from_deployment(deployment_id, api_key)` and
  `AstraClient(deployment_id, api_key)` need only the deployment id + API key.
- **CLI**: `astra pull | serve | bench` (`--base-url` is optional).
- Retries with exponential backoff on transient HTTP failures; telemetry can
  never raise into your serving path (drop-oldest queue, atexit flush).
- PEP 561 (`py.typed`), MIT license, full PyPI metadata.

## 0.1.0

- Initial release: `AstraClient.infer()` against the hosted
  `POST /api/v1/infer/{deployment_id}` endpoint.
