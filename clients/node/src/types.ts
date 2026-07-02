import type { OrtTensor } from "./ort.js";

export type TensorData =
  | number[]
  | bigint[]
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array;

/** One model input. `numpy`-less analogue of the Python SDK's name→ndarray. */
export interface TensorInput {
  data: TensorData;
  dims: number[];
  /** ONNX element type, e.g. "float32" (default), "int64", "uint8". */
  type?: string;
}

/** Inputs for a single inference: input-name → tensor. `null` synthesizes a
 *  random probe (smoke tests / benchmarks), matching `AstraRunner.run(None)`. */
export type RunInput = Record<string, TensorInput> | null | undefined;

export interface RunOutput {
  /** Inference time only (ms) — matches the Python SDK's `latencyMs`. */
  latencyMs: number;
  preMs: number;
  postMs: number;
  outputs: Array<{ name: string; shape: number[] }>;
  /** The raw onnxruntime output tensors (parity with Python's `raw`). */
  raw: OrtTensor[];
}

export interface ArtifactInfo {
  fileName: string;
  sizeBytes: number;
  sha256: string;
  kind?: string;
}
