// Tensor construction + output helpers. numpy-free analogue of AstraRunner's
// _prepare/_batch_of/_signature in the Python SDK.

import type { OrtModule, OrtTensor } from "./ort.js";
import type { TensorData, TensorInput } from "./types.js";

const FLOAT_TYPES = new Set(["float32", "float", "float16", "float64", "double"]);

export interface InputMeta {
  name: string;
  dims?: number[];
  type?: string;
}

function toBigInt(x: number | bigint): bigint {
  return typeof x === "bigint" ? x : BigInt(Math.trunc(Number(x)));
}

/** Coerce a JS array (or pass through a typed array) to the ONNX element type. */
function coerce(type: string, data: TensorData): OrtTensor["data"] {
  if (ArrayBuffer.isView(data)) return data as unknown as OrtTensor["data"];
  const arr = data as Array<number | bigint>;
  switch (type) {
    case "float32":
    case "float":
    case "float16":
      return Float32Array.from(arr as number[]);
    case "float64":
    case "double":
      return Float64Array.from(arr as number[]);
    case "int32":
      return Int32Array.from(arr as number[]);
    case "uint32":
      return Uint32Array.from(arr as number[]);
    case "int16":
      return Int16Array.from(arr as number[]);
    case "uint16":
      return Uint16Array.from(arr as number[]);
    case "int8":
      return Int8Array.from(arr as number[]);
    case "uint8":
    case "bool":
      return Uint8Array.from(arr as number[]);
    case "int64":
      return BigInt64Array.from(arr.map(toBigInt));
    case "uint64":
      return BigUint64Array.from(arr.map(toBigInt));
    default:
      return Float32Array.from(arr as number[]);
  }
}

/** Build an ort.Tensor for one input from the user-provided {data, dims, type}. */
export function buildTensor(
  ort: OrtModule,
  input: TensorInput,
  fallbackType = "float32",
): OrtTensor {
  const type = input.type ?? fallbackType;
  return new ort.Tensor(type, coerce(type, input.data) as never, input.dims);
}

function gaussian(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Synthesize a random valid probe per input (smoke tests / benchmarks). */
export function randomProbe(ort: OrtModule, metas: InputMeta[]): Record<string, OrtTensor> {
  const feeds: Record<string, OrtTensor> = {};
  for (const m of metas) {
    const dims = (m.dims ?? [1]).map((d) => (Number.isInteger(d) && d > 0 ? d : 1));
    const size = dims.reduce((a, b) => a * b, 1);
    const type = m.type ?? "float32";
    if (type === "int64" || type === "uint64") {
      const data =
        type === "int64" ? new BigInt64Array(size) : new BigUint64Array(size);
      feeds[m.name] = new ort.Tensor(type, data, dims);
    } else if (FLOAT_TYPES.has(type)) {
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) data[i] = gaussian();
      feeds[m.name] = new ort.Tensor("float32", data, dims);
    } else {
      feeds[m.name] = new ort.Tensor(type, coerce(type, new Array<number>(size).fill(0)) as never, dims);
    }
  }
  return feeds;
}

export function batchOf(feeds: Record<string, OrtTensor>): number {
  for (const t of Object.values(feeds)) {
    if (t.dims.length) return Number(t.dims[0]);
  }
  return 1;
}

export function signatureOf(feeds: Record<string, OrtTensor>): string | undefined {
  const parts: string[] = [];
  for (const [name, t] of Object.entries(feeds)) {
    parts.push(`${name}:${t.dims.join("x")}:${t.type}`);
  }
  return parts.length ? parts.join(";") : undefined;
}
