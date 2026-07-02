// AstraRunner — serve a Astra-deployed artifact on your own hardware.
// Mirrors clients/python/astra_sdk/runner.py.
//
//     import { AstraRunner } from "astra-ai-sdk";
//
//     const runner = await AstraRunner.fromDeployment({
//       baseUrl: "https://app.example.com",
//       deploymentId: "dep_ab12cd34ef",
//       apiKey: "astra_sk_live_…",
//     });
//     const out = await runner.run({ input: { data: myFloats, dims: [1, 3, 224, 224] } });
//     await runner.close();
//
// The artifact is pulled once via the API-key-authed
// GET /api/v1/artifacts/{deployment_id} and cached on disk keyed by its sha256,
// so restarts don't re-download. Every run() is measured (pre/infer/post) and
// shipped to the Astra dashboard by the background AstraTelemetryReporter.
//
// Requires onnxruntime-node:  npm i onnxruntime-node

import { existsSync, statSync } from "node:fs";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import { extname, join } from "node:path";

import { HttpSession, resolveBaseUrl } from "./http.js";
import { AstraTelemetryReporter } from "./telemetry.js";
import {
  batchOf,
  buildTensor,
  type InputMeta,
  randomProbe,
  signatureOf,
} from "./tensor.js";
import { VERSION } from "./version.js";
import type {
  OrtInferenceSession,
  OrtModule,
  OrtTensor,
  OrtValueMetadata,
} from "./ort.js";
import type { RunInput, RunOutput } from "./types.js";
import type { Sampleable } from "./stats.js";

const DEFAULT_CACHE = "~/.cache/astra";

export class AstraRunnerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AstraRunnerError";
  }
}

/** Thrown for caller mistakes (missing input) so run() classifies them as
 *  "bad_input" — matching the Python SDK's ValueError/KeyError handling. */
class BadInputError extends Error {}

let cachedOrt: OrtModule | null = null;

/** Lazily load onnxruntime-node; throw an actionable AstraRunnerError if absent. */
export async function requireServeExtra(): Promise<OrtModule> {
  if (cachedOrt) return cachedOrt;
  try {
    // Non-literal specifier: keeps TypeScript from trying to resolve the
    // optional native module at build time (it's loaded only at runtime).
    const spec = "onnxruntime-node";
    const mod = (await import(spec)) as { default?: OrtModule } & OrtModule;
    cachedOrt = (mod.default ?? mod) as OrtModule;
    return cachedOrt;
  } catch {
    throw new AstraRunnerError(
      "Local serving needs onnxruntime-node — install it: npm i onnxruntime-node",
    );
  }
}

function ortVersion(): string {
  try {
    const require = createRequire(import.meta.url);
    const pkg = require("onnxruntime-node/package.json") as { version?: string };
    return pkg.version ?? "";
  } catch {
    return "";
  }
}

// onnxruntime-node accepts short EP names ("coreml", "cuda") in
// executionProviders and listSupportedBackends() reports short names too. The
// dashboard groups Node and Python hosts by provider string, and the Python SDK
// reports full ORT names ("CoreMLExecutionProvider"), so normalize to the full
// name for telemetry — the session is still created with the caller's raw names.
const _EP_ALIASES: Record<string, string> = {
  cpu: "CPUExecutionProvider",
  coreml: "CoreMLExecutionProvider",
  cuda: "CUDAExecutionProvider",
  tensorrt: "TensorrtExecutionProvider",
  dml: "DmlExecutionProvider",
  directml: "DmlExecutionProvider",
  rocm: "ROCMExecutionProvider",
  webgpu: "WebGpuExecutionProvider",
  qnn: "QNNExecutionProvider",
  openvino: "OpenVINOExecutionProvider",
  nnapi: "NnapiExecutionProvider",
  xnnpack: "XnnpackExecutionProvider",
};

export function normalizeProvider(name: string): string {
  const n = name.trim();
  if (n.endsWith("ExecutionProvider")) return n;
  return _EP_ALIASES[n.toLowerCase()] ?? n;
}

/** The EP backends this onnxruntime-node build actually ships, as full ORT
 *  names. Empty when the build is too old to report them. */
export function supportedProviders(ort: OrtModule): string[] {
  try {
    const backends = ort.listSupportedBackends?.() ?? [];
    return backends.map((b) => normalizeProvider(b.name));
  } catch {
    return [];
  }
}

/** Resolve the provider identity reported in telemetry. `activeProvider` is the
 *  EP the session was asked to bind (the first requested, or CPU — the
 *  onnxruntime-node default); `availableProviders` is the honest list of EPs the
 *  build supports, so a CoreML/CUDA-capable host is never misreported as CPU-only. */
export function resolveProviders(
  ort: OrtModule,
  requested: string[] | undefined,
): { active: string; available: string[] } {
  const reqNorm = requested?.map(normalizeProvider).filter(Boolean);
  const active = reqNorm?.[0] ?? "CPUExecutionProvider";
  const supported = supportedProviders(ort);
  const available = supported.length ? supported : (reqNorm ?? [active]);
  // Ensure the bound EP is always present in the reported set.
  if (!available.includes(active)) available.unshift(active);
  return { active, available };
}

function expandHome(dir: string): string {
  if (dir === "~") return homedir();
  if (dir.startsWith("~/") || dir.startsWith("~\\")) {
    return join(homedir(), dir.slice(2));
  }
  return dir;
}

export interface PullArtifactOptions {
  deploymentId: string;
  apiKey: string;
  /** Optional — defaults to the hosted Astra origin (or ASTRA_BASE_URL). */
  baseUrl?: string;
  cacheDir?: string;
  timeout?: number;
}

/** Download the deployed artifact (sha256-cached under cacheDir). */
export async function pullArtifact(opts: PullArtifactOptions): Promise<string> {
  const { deploymentId, apiKey } = opts;
  const cacheDir = opts.cacheDir ?? DEFAULT_CACHE;
  const http = new HttpSession(resolveBaseUrl(opts.baseUrl), apiKey, {
    timeout: opts.timeout ?? 60,
  });
  try {
    const infoResp = await http.request("GET", `/api/v1/artifacts/${deploymentId}/info`);
    const info = (await infoResp.json()) as {
      fileName: string;
      sizeBytes: number;
      sha256: string;
    };
    const root = join(expandHome(cacheDir), deploymentId);
    await mkdir(root, { recursive: true });
    const suffix = extname(info.fileName) || ".onnx";
    const target = join(root, `${info.sha256}${suffix}`);
    if (existsSync(target) && statSync(target).size === info.sizeBytes) {
      return target;
    }
    const resp = await http.request("GET", `/api/v1/artifacts/${deploymentId}`);
    const buf = Buffer.from(await resp.arrayBuffer());
    const tmp = `${target}.part`;
    await writeFile(tmp, buf);
    await rename(tmp, target);
    await writeFile(join(root, "meta.json"), JSON.stringify(info, null, 2));
    return target;
  } finally {
    http.close();
  }
}

export interface FromDeploymentOptions {
  deploymentId: string;
  apiKey: string;
  /** Optional — defaults to the hosted Astra origin (or ASTRA_BASE_URL). */
  baseUrl?: string;
  cacheDir?: string;
  reportTelemetry?: boolean;
  providers?: string[];
  timeout?: number;
}

/** Local ONNX serving with built-in telemetry. */
export class AstraRunner {
  private reporter: AstraTelemetryReporter | null = null;
  private readonly metas: InputMeta[];
  private readonly activeProviderName: string;

  private constructor(
    private readonly ort: OrtModule,
    private readonly session: OrtInferenceSession,
    activeProvider: string,
  ) {
    this.activeProviderName = activeProvider;
    this.metas = this.readInputMetas();
  }

  static async fromDeployment(opts: FromDeploymentOptions): Promise<AstraRunner> {
    const ort = await requireServeExtra();
    const base = resolveBaseUrl(opts.baseUrl);
    const path = await pullArtifact({
      baseUrl: base,
      deploymentId: opts.deploymentId,
      apiKey: opts.apiKey,
      cacheDir: opts.cacheDir,
      timeout: opts.timeout ?? 60,
    });
    const session = await ort.InferenceSession.create(
      path,
      opts.providers ? { executionProviders: opts.providers } : undefined,
    );
    // onnxruntime-node does not expose the EP it actually bound, so record the
    // requested provider (or CPU) as active — but report the build's real
    // supported EPs so a GPU/NPU-capable host isn't misreported as CPU-only.
    const { active, available } = resolveProviders(ort, opts.providers);
    const runner = new AstraRunner(ort, session, active);
    runner.reporter = new AstraTelemetryReporter(base, opts.deploymentId, opts.apiKey, {
      sdkVersion: VERSION,
      enabled: opts.reportTelemetry ?? true,
      activeProvider: active,
      ortVersion: ortVersion(),
      availableProviders: available,
      // Land the durable telemetry spool under the same cache root as the
      // artifact (~/.cache/astra/<deployment>/telemetry/ by default).
      cacheDir: opts.cacheDir,
    });
    return runner;
  }

  /**
   * Serve a model file you ALREADY have on disk — e.g. the one from the SDK Hub
   * "Download Artifact" button, or an artifact committed next to your code. No
   * deployment / network needed. Telemetry is OFF unless you also pass a
   * deployment id + API key (then local runs still report to that deployment).
   */
  static async fromFile(
    modelPath: string,
    opts: {
      providers?: string[];
      deploymentId?: string;
      apiKey?: string;
      baseUrl?: string;
      reportTelemetry?: boolean;
    } = {},
  ): Promise<AstraRunner> {
    const ort = await requireServeExtra();
    const session = await ort.InferenceSession.create(
      modelPath,
      opts.providers ? { executionProviders: opts.providers } : undefined,
    );
    const { active, available } = resolveProviders(ort, opts.providers);
    const runner = new AstraRunner(ort, session, active);
    // Report telemetry only when a deployment is supplied — there's nowhere to
    // send it otherwise. Defaults on in that case, off for a bare file.
    const wantTelemetry =
      opts.reportTelemetry ?? Boolean(opts.deploymentId && opts.apiKey);
    if (wantTelemetry && opts.deploymentId && opts.apiKey) {
      runner.reporter = new AstraTelemetryReporter(
        resolveBaseUrl(opts.baseUrl),
        opts.deploymentId,
        opts.apiKey,
        {
          sdkVersion: VERSION,
          enabled: true,
          activeProvider: active,
          ortVersion: ortVersion(),
          availableProviders: available,
        },
      );
    }
    return runner;
  }

  get activeProvider(): string {
    return this.activeProviderName;
  }

  /** The model's input specs (name, dims, type) — handy for building probes. */
  inputSpecs(): InputMeta[] {
    return this.metas.map((m) => ({
      name: m.name,
      dims: m.dims ? [...m.dims] : undefined,
      type: m.type,
    }));
  }

  private readInputMetas(): InputMeta[] {
    const meta = (this.session as { inputMetadata?: ReadonlyArray<OrtValueMetadata> })
      .inputMetadata;
    if (Array.isArray(meta) && meta.length) {
      return meta.map((m) => ({
        name: m.name,
        dims: normDims(m.dimensions ?? m.shape),
        type: normType(m.type),
      }));
    }
    return this.session.inputNames.map((name) => ({ name }));
  }

  async run(inputs: RunInput = null, opts: { region?: string } = {}): Promise<RunOutput> {
    const region = opts.region ?? "local";
    const t0 = performance.now();
    let feeds: Record<string, OrtTensor> = {};
    try {
      feeds = this.prepare(inputs);
      const preMs = performance.now() - t0;

      const t1 = performance.now();
      const result = await this.session.run(feeds);
      const inferMs = performance.now() - t1;

      const t2 = performance.now();
      const outputs: Array<{ name: string; shape: number[] }> = [];
      const raw: OrtTensor[] = [];
      for (const name of this.session.outputNames) {
        const t = result[name];
        if (!t) continue;
        raw.push(t);
        outputs.push({ name, shape: [...t.dims] });
      }
      const first = raw[0] ?? null;
      this.reporter?.observe(
        feedsToSampleable(feeds),
        first ? { data: first.data as ArrayLike<number | bigint>, dims: first.dims } : null,
      );
      const postMs = performance.now() - t2;

      this.reporter?.recordEvent({
        latencyMs: inferMs,
        preMs,
        postMs,
        success: true,
        batchSize: batchOf(feeds),
        region,
        inputSig: signatureOf(feeds),
      });
      return {
        latencyMs: round3(inferMs),
        preMs: round3(preMs),
        postMs: round3(postMs),
        outputs,
        raw,
      };
    } catch (err) {
      const errorCode = err instanceof BadInputError ? "bad_input" : "inference_error";
      this.reporter?.recordEvent({
        latencyMs: 0,
        success: false,
        errorCode,
        batchSize: batchOf(feeds),
        region,
        inputSig: signatureOf(feeds),
      });
      throw err;
    }
  }

  private prepare(inputs: RunInput): Record<string, OrtTensor> {
    if (inputs == null) {
      return randomProbe(this.ort, this.metas);
    }
    const feeds: Record<string, OrtTensor> = {};
    for (const name of this.session.inputNames) {
      const userInput = inputs[name];
      if (!userInput) throw new BadInputError(`missing input '${name}'`);
      feeds[name] = buildTensor(this.ort, userInput);
    }
    return feeds;
  }

  async close(): Promise<void> {
    await this.reporter?.close();
    try {
      await (this.session as { release?: () => Promise<void> }).release?.();
    } catch {
      /* best effort */
    }
  }
}

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

function feedsToSampleable(feeds: Record<string, OrtTensor>): Record<string, Sampleable> {
  const out: Record<string, Sampleable> = {};
  for (const [k, t] of Object.entries(feeds)) {
    out[k] = { data: t.data as ArrayLike<number | bigint>, dims: t.dims };
  }
  return out;
}

function normDims(shape: ReadonlyArray<number | string> | undefined): number[] | undefined {
  if (!shape) return undefined;
  // Symbolic/dynamic dims (strings, <=0) become -1; randomProbe maps them to 1.
  return shape.map((d) => (typeof d === "number" && Number.isInteger(d) && d > 0 ? d : -1));
}

function normType(type: string | undefined): string | undefined {
  if (!type) return undefined;
  // ORT metadata types look like "tensor(float)" / "tensor(int64)".
  const m = /tensor\(([^)]+)\)/.exec(type);
  const t = (m ? m[1] : type)!.toLowerCase();
  switch (t) {
    case "float":
      return "float32";
    case "double":
      return "float64";
    default:
      return t;
  }
}
