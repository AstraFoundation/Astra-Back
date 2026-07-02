import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import type { OrtModule, OrtTensor } from "../src/ort.js";
import {
  AstraRunner,
  normalizeProvider,
  pullArtifact,
  requireServeExtra,
  resolveProviders,
} from "../src/runner.js";
import { buildTensor } from "../src/tensor.js";
import { installFetch } from "./_mock.js";

// A fake ort module whose Tensor just records (type, data, dims) — lets us test
// dtype-aware tensor construction without the native binary.
const fakeOrt = {
  Tensor: class {
    constructor(
      public type: string,
      public data: unknown,
      public dims: readonly number[],
    ) {}
  },
} as unknown as OrtModule;

describe("pullArtifact", () => {
  it("downloads, sha256-caches, and reuses without re-downloading", async () => {
    const payload = Buffer.from("fake-onnx-bytes");
    const { calls } = installFetch((req) => {
      if (req.path.endsWith("/info")) {
        return {
          json: { fileName: "model.onnx", sizeBytes: payload.length, sha256: "abc123" },
        };
      }
      return { bytes: new Uint8Array(payload) };
    });
    const cacheDir = await mkdtemp(join(tmpdir(), "astra-test-"));

    const path1 = await pullArtifact({
      baseUrl: "http://t",
      deploymentId: "dep_x",
      apiKey: "k",
      cacheDir,
    });
    expect(path1.endsWith("abc123.onnx")).toBe(true);
    expect(await readFile(path1)).toEqual(payload);
    expect(calls).toHaveLength(2); // /info + /artifacts download

    const before = calls.length;
    const path2 = await pullArtifact({
      baseUrl: "http://t",
      deploymentId: "dep_x",
      apiKey: "k",
      cacheDir,
    });
    expect(path2).toBe(path1);
    expect(calls.length - before).toBe(1); // only /info; bytes served from cache
  });
});

describe("requireServeExtra", () => {
  it("throws a AstraRunnerError when onnxruntime-node is absent", async () => {
    let present = true;
    try {
      await import("onnxruntime-node");
    } catch {
      present = false;
    }
    if (present) return; // skip if the optional dep happens to be installed
    await expect(requireServeExtra()).rejects.toMatchObject({ name: "AstraRunnerError" });
  });
});

describe("AstraRunner.fromFile", () => {
  it("is a static factory that rejects a missing model path", async () => {
    expect(typeof AstraRunner.fromFile).toBe("function");
    await expect(AstraRunner.fromFile("/no/such/model.onnx")).rejects.toBeTruthy();
  });
});

describe("provider reporting (honest availableProviders)", () => {
  // A build that ships CPU + CoreML + WebGPU (what onnxruntime-node reports on a
  // Mac) — listSupportedBackends uses short names, like the real module.
  const ortWithCoreml = {
    listSupportedBackends: () => [
      { name: "cpu", bundled: true },
      { name: "coreml", bundled: true },
      { name: "webgpu", bundled: true },
    ],
  } as unknown as OrtModule;

  it("normalizes short EP names to full ORT names (Python parity)", () => {
    expect(normalizeProvider("coreml")).toBe("CoreMLExecutionProvider");
    expect(normalizeProvider("cuda")).toBe("CUDAExecutionProvider");
    expect(normalizeProvider("qnn")).toBe("QNNExecutionProvider");
    // Already-full names pass through unchanged; unknown names too.
    expect(normalizeProvider("CoreMLExecutionProvider")).toBe("CoreMLExecutionProvider");
    expect(normalizeProvider("SomethingNew")).toBe("SomethingNew");
  });

  it("reports the build's real supported EPs, not CPU-only, when none requested", () => {
    const { active, available } = resolveProviders(ortWithCoreml, undefined);
    // Bound EP defaults to CPU (onnxruntime-node's default) — honest.
    expect(active).toBe("CPUExecutionProvider");
    // But a CoreML-capable host must NOT be misreported as CPU-only.
    expect(available).toContain("CoreMLExecutionProvider");
    expect(available).toContain("CPUExecutionProvider");
  });

  it("honors an explicitly requested provider and always includes it", () => {
    const { active, available } = resolveProviders(ortWithCoreml, ["coreml"]);
    expect(active).toBe("CoreMLExecutionProvider");
    expect(available).toContain("CoreMLExecutionProvider");
  });

  it("falls back to [active] when the build can't list backends", () => {
    const bareOrt = {} as unknown as OrtModule;
    const { active, available } = resolveProviders(bareOrt, ["cuda"]);
    expect(active).toBe("CUDAExecutionProvider");
    expect(available).toEqual(["CUDAExecutionProvider"]);
  });
});

describe("buildTensor", () => {
  it("maps a float input to Float32Array by default", () => {
    const t = buildTensor(fakeOrt, { data: [1, 2, 3], dims: [3] }) as OrtTensor & {
      data: unknown;
    };
    expect(t.type).toBe("float32");
    expect(t.data).toBeInstanceOf(Float32Array);
  });

  it("maps an int64 input to BigInt64Array (non-float dtype)", () => {
    const t = buildTensor(fakeOrt, {
      data: [1, 2, 3],
      dims: [3],
      type: "int64",
    }) as OrtTensor & { data: BigInt64Array };
    expect(t.type).toBe("int64");
    expect(t.data).toBeInstanceOf(BigInt64Array);
    expect(t.data[0]).toBe(1n);
  });
});
