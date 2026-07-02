// Minimal structural typing of the onnxruntime-node surface this SDK uses, so
// typecheck and build never require the optional native binary to be present.
// At runtime the real module is loaded via dynamic `import("onnxruntime-node")`
// (see runner.ts) and structurally satisfies these interfaces.

export interface OrtTensor {
  readonly dims: readonly number[];
  readonly type: string;
  readonly data: ArrayLike<number> | ArrayLike<bigint> | readonly string[];
  readonly size?: number;
}

export type OrtTensorData =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array
  | number[]
  | bigint[]
  | string[];

export interface OrtTensorConstructor {
  new (type: string, data: OrtTensorData, dims: readonly number[]): OrtTensor;
}

export interface OrtValueMetadata {
  readonly name: string;
  // ONNX Runtime exposes input metadata inconsistently across versions; these
  // are read defensively (runner.ts) and absence is treated as "unknown shape".
  readonly type?: string;
  readonly shape?: ReadonlyArray<number | string>;
  readonly dimensions?: ReadonlyArray<number | string>;
  readonly isTensor?: boolean;
}

export interface OrtInferenceSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  readonly inputMetadata?: ReadonlyArray<OrtValueMetadata>;
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

export interface OrtSessionOptions {
  executionProviders?: string[];
}

export interface OrtModule {
  Tensor: OrtTensorConstructor;
  InferenceSession: {
    create(
      path: string,
      options?: OrtSessionOptions,
    ): Promise<OrtInferenceSession>;
  };
  // The execution-provider backends compiled into this onnxruntime-node build
  // (e.g. [{name:"cpu"}, {name:"coreml"}, {name:"webgpu"}]). Present since
  // onnxruntime-node 1.16; read defensively (may be absent on older builds).
  listSupportedBackends?: () => ReadonlyArray<{ name: string; bundled?: boolean }>;
}
