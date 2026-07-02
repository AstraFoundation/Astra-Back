// astra-ai-sdk — run Astra-compressed models ON-DEVICE, with a closed telemetry loop.
//
// Astra never runs your model server-side. The SDK pulls the compressed artifact
// once (npm i onnxruntime-node) and serves it on YOUR hardware:
//
//     import { AstraRunner } from "astra-ai-sdk";
//     const runner = await AstraRunner.fromDeployment({ deploymentId, apiKey });
//     const out = await runner.run({ input: { data: myFloats, dims: [1, 3, 224, 224] } });
//     await runner.close();
//
// Closed-loop telemetry buffers durably on disk while offline and flushes to the
// dashboard on reconnect, then deletes after the server acks. Opt out with
// reportTelemetry:false / ASTRA_SDK_TELEMETRY=0; disable disk buffering with
// ASTRA_SDK_SPOOL=0.

export { AstraApiError } from "./http.js";
export { AstraRunner, pullArtifact, requireServeExtra, AstraRunnerError } from "./runner.js";
export { AstraTelemetryReporter, telemetryEnabled } from "./telemetry.js";
export { VERSION } from "./version.js";

export type { FromDeploymentOptions, PullArtifactOptions } from "./runner.js";
export type { InputMeta } from "./tensor.js";
export type { RecordEventOptions, TelemetryReporterOptions } from "./telemetry.js";
export type { ArtifactInfo, RunInput, RunOutput, TensorData, TensorInput } from "./types.js";
