# Changelog

Versions are kept in **lockstep** with the Python `astra-ai-sdk` so the dashboard's
single `GET /api/sdk/version` describes both clients truthfully.

## Unreleased

- **Fix — honest execution-provider reporting.** Telemetry snapshots now report
  `availableProviders` from onnxruntime-node's `listSupportedBackends()` (the EPs
  the build actually ships, e.g. `CoreMLExecutionProvider`) instead of echoing
  only the requested provider, and normalize short EP names (`coreml`, `cuda`) to
  their full ORT names so Node and Python hosts on the same accelerator group
  together on the dashboard. A CoreML/CUDA-capable host is no longer misreported
  as CPU-only. `activeProvider` is unchanged (the requested EP, or CPU default —
  onnxruntime-node does not expose the bound EP).

## 0.4.0

- **Breaking — public classes rebranded to the Astra namespace.** `LocalRunner`
  → **`AstraRunner`**, `RunnerError` → **`AstraRunnerError`**, `TelemetryReporter`
  → **`AstraTelemetryReporter`**, `ApiError` → **`AstraApiError`**. Update imports:
  `import { AstraRunner } from "astra-ai-sdk"`. The package name, the `astra` CLI,
  function/constant names, and all behavior are unchanged.

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

## 0.2.0 — 2026-06-29

- Initial Node release — API parity with the Python `astra-ai-sdk` 0.2.0.
- **No base URL in your code**: `baseUrl` is optional everywhere — it defaults to
  the hosted Astra origin (override with `ASTRA_BASE_URL` or the `baseUrl`
  option). SDK code only needs the deployment id + API key. `AstraClient` is now
  `new AstraClient(deploymentId, apiKey, { baseUrl? })`.
- **Hosted inference**: `AstraClient.infer()` against
  `POST /api/v1/infer/{deployment_id}` (zero runtime dependencies — uses the
  global `fetch`).
- **Local serving**: `LocalRunner.fromDeployment()` pulls the deployed,
  Astra-compressed artifact (sha256-cached on disk) and runs it with
  onnxruntime-node **inside your own code** — no server to stand up.
- **Run a local file**: `LocalRunner.fromFile("model.onnx")` serves an artifact
  you already have (e.g. the SDK Hub "Download Artifact" file) — no deployment
  needed; telemetry off unless you pass `{ deploymentId, apiKey }`.
- **Built-in telemetry**: every local inference is measured (latency breakdown,
  batch, input signature) and shipped in fault-tolerant background batches to
  the Astra dashboard, plus periodic system snapshots and windowed input/output
  distribution stats that power prediction/input drift alerts. Opt out with
  `reportTelemetry: false` or `ASTRA_SDK_TELEMETRY=0`.
- **CLI**: `astra pull | serve | bench`.
- Retries with exponential backoff on transient HTTP failures; telemetry can
  never raise into your serving path (drop-oldest queue, `beforeExit` flush).
- `onnxruntime-node` is an optional dependency; ESM-only; ships `.d.ts` types.
