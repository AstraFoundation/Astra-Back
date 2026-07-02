// SDK e2e (Node) — runs with the TARBALL-INSTALLED astra-ai-sdk from a /tmp project.
//
// Pulls the deployed artifact, serves it locally with AstraRunner, fires normal
// traffic then distribution-shifted traffic, and flushes telemetry. No repo
// imports — only the installed package + onnxruntime-node. Mirrors the Python
// _sdk_e2e_client.py so the shared _sdk_e2e_assert.py works unchanged.

import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: { handoff: { type: "string" } } });
const h = JSON.parse(readFileSync(values.handoff, "utf8"));

const { AstraRunner } = await import("astra-ai-sdk");

// Prove we imported the INSTALLED tarball, not the repo source.
const resolved = import.meta.resolve("astra-ai-sdk");
if (!resolved.includes("/node_modules/astra-ai-sdk/")) {
  throw new Error(`must import the INSTALLED tarball, got ${resolved}`);
}
console.log(`   using ${resolved}`);

const runner = await AstraRunner.fromDeployment({
  baseUrl: h.baseUrl,
  deploymentId: h.deploymentId,
  apiKey: h.apiKey,
  cacheDir: "/tmp/astra-node-cache",
});

const spec = runner.inputSpecs()[0];
const name = spec.name;
const dims = (spec.dims ?? [1]).map((d) => (Number.isInteger(d) && d > 0 ? d : 1));
const size = dims.reduce((a, b) => a * b, 1);

function gauss() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function probe(shift = 0) {
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) data[i] = gauss() + shift;
  return { [name]: { data, dims, type: "float32" } };
}

const t0 = Date.now();
for (let i = 0; i < 100; i++) await runner.run(probe());
console.log(`   100 baseline inferences in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

// Distribution shift: +5 sigma on every input feature.
for (let i = 0; i < 60; i++) await runner.run(probe(5.0));
console.log("   60 shifted inferences (input mean +5.0)");

await runner.close(); // final telemetry flush (beforeExit budget)
console.log("   telemetry flushed");
