import { defineConfig } from "tsup";

// Two entries: the library (index) and the `astra` CLI. The CLI keeps its
// `#!/usr/bin/env node` shebang from src/cli.ts (esbuild preserves the entry
// shebang), so no banner is needed. ESM-only — the repo is ESM and the SDK
// relies on dynamic `import()` of the optional onnxruntime-node dependency.
export default defineConfig({
  entry: ["src/index.ts", "src/cli.ts"],
  format: ["esm"],
  target: "node18",
  platform: "node",
  dts: true,
  clean: true,
  sourcemap: true,
  // onnxruntime-node is an optional peer; never bundle it — it's loaded at
  // runtime via dynamic import only when AstraRunner is used.
  external: ["onnxruntime-node"],
});
